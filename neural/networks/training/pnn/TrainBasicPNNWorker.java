package neural.networks.training.pnn;

import java.util.concurrent.Callable;

import org.encog.util.EngineArray;
import neural.pnn.BasicPNN;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import neural.pnn.PNNKernelType;
import neural.pnn.PNNOutputMode;

public class TrainBasicPNNWorker implements Callable<TrainBasicPNNWorkerResult> {

	private int from;
	private int to;
	private double error, totalError;
	private double diff;
	private BasicPNN network;
	private MLDataSet training;
	private boolean useDeriv;

	/**
	 * Temp storage for derivative computation. doesn't use value from earlier
	 * derivative calls so not a problem for concurrency
	 */
	private double[] v;

	/**
	 * Temp storage for derivative computation. doesn't use value from earlier
	 * derivative calls so not a problem for concurrency
	 */
	private double[] w;

	/**
	 * Temp storage for derivative computation. doesn't use value from earlier
	 * derivative calls so not a problem for concurrency
	 */
	private double[] dsqr;

	private double[] deriv;

	private double[] deriv2;

	private double[] sigma;

	public TrainBasicPNNWorker(final BasicPNN network, final MLDataSet training, final boolean useDeriv,
			final double[] deriv, final double[] deriv2, final double[] sigma, final int from, final int to) {
		this.network = network;
		this.training = training;
		this.useDeriv = useDeriv;
		this.deriv = deriv;
		// Since sigma is not expected to change during training copying might
		// not be needed but lets do it to be safe
		this.sigma = new double[sigma.length];
		EngineArray.arrayCopy(sigma, this.sigma);
		v = null;
		w = null;
		dsqr = null;

		if (useDeriv) {
			this.deriv = new double[deriv.length];
			EngineArray.arrayCopy(deriv, this.deriv);
			this.deriv2 = new double[deriv2.length];
			EngineArray.arrayCopy(deriv2, this.deriv2);
			int k;

			if (this.network.getOutputMode() == PNNOutputMode.Classification) {
				k = this.network.getOutputCount();
			} else {
				k = this.network.getOutputCount() + 1;
			}

			this.dsqr = new double[this.network.getInputCount()];
			this.v = new double[this.network.getInputCount() * k];
			this.w = new double[this.network.getInputCount() * k];
		}
		this.from = from;
		this.to = to;
	}

	public TrainBasicPNNWorkerResult call() {
		this.totalError = calculateError();
		TrainBasicPNNWorkerResult result = new TrainBasicPNNWorkerResult(totalError);
		if (useDeriv) {
			result.setDeriv(deriv);
			result.setDeriv2(deriv2);
		}
		return result;
	}

	protected double calculateError() {
		this.totalError = 0.0;
		if (useDeriv) {
			final int num = (this.network.isSeparateClass())
					? this.network.getInputCount() * this.network.getOutputCount() : this.network.getInputCount();

			this.deriv = new double[num];
			this.deriv2 = new double[num];
			for (int i = 0; i < num; i++) {
				this.deriv[i] = 0.0;
				this.deriv2[i] = 0.0;
			}
		}

		final MLDataPair pair = BasicMLDataPair.createPair(training.getInputSize(), training.getIdealSize());

		final double[] out = new double[this.network.getOutputCount()];

		for (int r = from; r <= to; r++) {
			training.getRecord(r, pair);

			error = 0.0;

			final MLData input = pair.getInput();
			final MLData target = pair.getIdeal();

			// CLASSIFICATION
			{
				final int tclass = (int) target.getData(0);
				MLData output;

				if (useDeriv)
					if (this.network.isSeparateClass())
						output = computeDerivSepClass(input, pair.getIdeal(), this.sigma);
					else
						output = computeDeriv(input, pair.getIdeal(), this.sigma);
				else {
					output = this.network.compute(input);
				}

				EngineArray.arrayCopy(output.getData(), out);

				for (int i = 0; i < out.length; i++) {
					if (i == tclass) {
						diff = 1.0 - out[i];
						error += diff * diff;
					} else {
						error += out[i] * out[i];
					}
				}
			}
			totalError += error;
		}
		return totalError;
	}

	/**
	 * Compute the derivative for target data.
	 *
	 * the global variables v w dsqr are independent of earlier calls while
	 * deriv1 & deriv2 are not so their magnitude will be affected of the subset
	 * size maxCompares when used
	 * 
	 * @param input
	 *            The input.
	 * @param target
	 *            The target data.
	 * @return The output.
	 */

	private MLData computeDeriv(final MLData input, final MLData target, double[] sigma) {
		int pop, ivar;
		int outvar;
		double diff, dist, truedist;
		double vtot, wtot;
		double temp, der1, der2;
		int vptr, wptr, vsptr = 0, wsptr = 0;
		final double[] out = new double[this.network.getOutputCount()];

		for (pop = 0; pop < this.network.getOutputCount(); pop++) {
			out[pop] = 0.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				this.v[pop * this.network.getInputCount() + ivar] = 0.0;
				this.w[pop * this.network.getInputCount() + ivar] = 0.0;
			}
		}

		if (this.network.getOutputMode() != PNNOutputMode.Classification) {
			vsptr = this.network.getOutputCount() * this.network.getInputCount();
			wsptr = this.network.getOutputCount() * this.network.getInputCount();
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				this.v[vsptr + ivar] = 0.0;
				this.w[wsptr + ivar] = 0.0;
			}
		}

		final MLDataPair pair = BasicMLDataPair.createPair(this.network.getSamples().getInputSize(),
				this.network.getSamples().getIdealSize());

		int nrOfSamples = (int) this.network.getSamples().getRecordCount();
		// int compares = 0;
		double psum = 0.0;

		for (int r = 0; r < nrOfSamples; r++) {
			MLDataPair dataPair = this.training.get(r);
			pair.setIdealArray(dataPair.getIdealArray());
			pair.setInputArray(dataPair.getInputArray());

			if (BasicPNN.isSamePattern(input, pair.getInput())) {
				continue;
			}

			dist = 0.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				diff = input.getData(ivar) - pair.getInput().getData(ivar);
				diff /= sigma[ivar];
				this.dsqr[ivar] = diff * diff;
				dist += this.dsqr[ivar];
			}

			if (this.network.getKernel() == PNNKernelType.Gaussian) {
				dist = Math.exp(-dist);
			} else if (this.network.getKernel() == PNNKernelType.Reciprocal) {
				dist = 1.0 / (1.0 + dist);
			}

			truedist = dist;
			if (dist < 1.e-40) {
				dist = 1.e-40;
			}

			if (this.network.getOutputMode() == PNNOutputMode.Classification) {
				pop = (int) pair.getIdeal().getData(0);
				out[pop] += dist;
				vptr = pop * this.network.getInputCount();
				wptr = pop * this.network.getInputCount();
				for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
					temp = truedist * this.dsqr[ivar];
					this.v[vptr + ivar] += temp;
					this.w[wptr + ivar] += temp * (2.0 * this.dsqr[ivar] - 3.0);
				}
			}

			else if (this.network.getOutputMode() == PNNOutputMode.Unsupervised) {
				for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
					out[ivar] += dist * pair.getInput().getData(ivar);
					temp = truedist * this.dsqr[ivar];
					this.v[vsptr + ivar] += temp;
					this.w[wsptr + ivar] += temp * (2.0 * this.dsqr[ivar] - 3.0);
				}
				vptr = 0;
				wptr = 0;
				for (outvar = 0; outvar < this.network.getOutputCount(); outvar++) {
					for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
						temp = truedist * this.dsqr[ivar] * pair.getInput().getData(ivar);
						this.v[vptr++] += temp;
						this.w[wptr++] += temp * (2.0 * this.dsqr[ivar] - 3.0);
					}
				}
				psum += dist;
			} else if (this.network.getOutputMode() == PNNOutputMode.Regression) {

				for (ivar = 0; ivar < this.network.getOutputCount(); ivar++) {
					out[ivar] += dist * pair.getIdeal().getData(ivar);
				}
				vptr = 0;
				wptr = 0;
				for (outvar = 0; outvar < this.network.getOutputCount(); outvar++) {
					for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
						temp = truedist * this.dsqr[ivar] * pair.getIdeal().getData(outvar);
						this.v[vptr++] += temp;
						this.w[wptr++] += temp * (2.0 * this.dsqr[ivar] - 3.0);
					}
				}
				for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
					temp = truedist * this.dsqr[ivar];
					this.v[vsptr + ivar] += temp;
					this.w[wsptr + ivar] += temp * (2.0 * this.dsqr[ivar] - 3.0);
				}

				psum += dist;
				// compares++;
			}
		}

		// v and w is now updated for this pattern

		if (this.network.getOutputMode() == PNNOutputMode.Classification) {
			psum = 0.0;
			for (pop = 0; pop < this.network.getOutputCount(); pop++) {
				if (this.network.usePriors()) {
					out[pop] *= ((double) this.network.getOutputCount()) / this.network.getPriors()[pop];
				}
				psum += out[pop];
			}

			if (psum < 1.e-40) {
				psum = 1.e-40;
			}
		}

		for (pop = 0; pop < this.network.getOutputCount(); pop++) {
			out[pop] /= psum;
		}

		// nested loop ivar and outvar
		for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
			if (this.network.getOutputMode() == PNNOutputMode.Classification) {
				vtot = wtot = 0.0;
			} else {
				vtot = this.v[vsptr + ivar] * 2.0 / (psum * this.network.getSigma()[ivar]);
				wtot = this.w[wsptr + ivar] * 2.0
						/ (psum * this.network.getSigma()[ivar] * this.network.getSigma()[ivar]);
			}

			for (outvar = 0; outvar < this.network.getOutputCount(); outvar++) {
				if ((this.network.getOutputMode() == PNNOutputMode.Classification) && (this.network.usePriors())) {
					this.v[outvar * this.network.getInputCount() + ivar] *= ((double) this.network.getOutputCount())
							/ this.network.getPriors()[outvar];
					this.w[outvar * this.network.getInputCount() + ivar] *= this.v[outvar * this.network.getInputCount()
							+ ivar];
				}
				this.v[outvar * this.network.getInputCount() + ivar] *= 2.0 / (psum * this.network.getSigma()[ivar]);
				this.w[outvar * this.network.getInputCount() + ivar] *= 2.0
						/ (psum * this.network.getSigma()[ivar] * this.network.getSigma()[ivar]);
				if (this.network.getOutputMode() == PNNOutputMode.Classification) {
					vtot += this.v[outvar * this.network.getInputCount() + ivar];
					wtot += this.w[outvar * this.network.getInputCount() + ivar];
				}
			}

			for (outvar = 0; outvar < this.network.getOutputCount(); outvar++) {
				der1 = this.v[outvar * this.network.getInputCount() + ivar] - out[outvar] * vtot;
				der2 = this.w[outvar * this.network.getInputCount() + ivar] + 2.0 * out[outvar] * vtot * vtot
						- 2.0 * this.v[outvar * this.network.getInputCount() + ivar] * vtot - out[outvar] * wtot;
				if (this.network.getOutputMode() == PNNOutputMode.Classification) {

					if (outvar == (int) target.getData(0)) {
						temp = 2.0 * (out[outvar] - 1.0);
					} else {
						temp = 2.0 * out[outvar];
					}
				} else {
					temp = 2.0 * (out[outvar] - target.getData(outvar));
				}

				this.deriv[ivar] += temp * der1;
				this.deriv2[ivar] += temp * der2 + 2.0 * der1 * der1;
			}
		}
		return new BasicMLData(out);
	}

	/**
	 * Compute the derivative for target data when using separate sigmas for
	 * each target class.
	 * 
	 * @param input
	 *            The input.
	 * @param target
	 *            The target data.
	 * @return The output.
	 */
	private MLData computeDerivSepClass(final MLData input, final MLData target, double[] sigma) {
		int pop, ivar;
		int outvar;
		double diff, dist, truedist;
		double temp, der1, der2, psum;
		int vptr, wptr;
		final double[] out = new double[this.network.getOutputCount()];
		for (pop = 0; pop < this.network.getOutputCount(); pop++) {
			out[pop] = 0.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				this.v[pop * this.network.getInputCount() + ivar] = 0.0;
				this.w[pop * this.network.getInputCount() + ivar] = 0.0;
			}
		}

		final MLDataPair pair = BasicMLDataPair.createPair(this.network.getSamples().getInputSize(),
				this.network.getSamples().getIdealSize());

		int nrOfSamples = (int) this.network.getSamples().getRecordCount();

		for (int r = 0; r < nrOfSamples; r++) {
			this.network.getSamples().getRecord(r, pair);
			if (BasicPNN.isSamePattern(input, pair.getInput())) {
				continue;
			}
			pop = (int) pair.getIdeal().getData(0);
			dist = 0.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				diff = input.getData(ivar) - pair.getInput().getData(ivar);
				diff /= sigma[pop * this.network.getInputCount() + ivar];
				this.dsqr[ivar] = diff * diff;
				dist += this.dsqr[ivar];
			}

			if (this.network.getKernel() == PNNKernelType.Gaussian) {
				dist = Math.exp(-dist);
			} else if (this.network.getKernel() == PNNKernelType.Reciprocal) {
				dist = 1.0 / (1.0 + dist);
			}

			truedist = dist;
			if (dist < 1.e-40) {
				dist = 1.e-40;
			}

			out[pop] += dist;
			vptr = pop * this.network.getInputCount();
			wptr = pop * this.network.getInputCount();
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				temp = truedist * this.dsqr[ivar];
				this.v[vptr + ivar] += temp;
				this.w[wptr + ivar] += temp * (2.0 * this.dsqr[ivar] - 3.0);
			}
		} // For all training cases

		/*
		 * Scale the the outputs per the sigmas Make the v and w be the actual
		 * derivatives of the activations if Classification only ?
		 */
		for (outvar = 0; outvar < this.network.getOutputCount(); outvar++) {
			temp = 1.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				temp *= sigma[outvar * this.network.getInputCount() + ivar];
			}
			out[outvar] /= temp;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				int sigmaPtr = outvar * this.network.getInputCount() + ivar;
				this.v[sigmaPtr] *= 2.0 / sigma[sigmaPtr];
				this.w[sigmaPtr] *= 2.0 / (sigma[sigmaPtr] * sigma[sigmaPtr]);
				// v and w is now updated for this pattern
				this.v[sigmaPtr] /= temp;// Scale first and
				this.w[sigmaPtr] /= temp;// second derivatives
				// Apply equations 5.27 5.28
				this.w[sigmaPtr] += 2.0 / sigma[sigmaPtr] * (out[outvar] / sigma[sigmaPtr] - v[sigmaPtr]);
				this.v[sigmaPtr] -= out[outvar] / sigma[sigmaPtr];
			}
		}

		/*
		 * Deal with class count normalization and prior probabilities.
		 */

		psum = 0.0;
		for (pop = 0; pop < this.network.getOutputCount(); pop++) {
			if (this.network.getPriors()[pop] >= 0.0) {
				out[pop] *= this.network.getPriors()[pop] / this.network.getCountPer()[pop];
			}
			psum += out[pop];
		}

		if (psum < 1.e-40) {
			psum = 1.e-40;
		}

		for (pop = 0; pop < this.network.getOutputCount(); pop++) {
			out[pop] /= psum;
		} // out[pop] done

		/*
		 * Compute the derivatives nested loop ivar and outvar
		 */
		for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
			for (outvar = 0; outvar < this.network.getOutputCount(); outvar++)
			// Apply priors to derivs
			{
				int sigmaPtr = outvar * this.network.getInputCount() + ivar;
				if (this.network.usePriors()) {
					this.v[sigmaPtr] *= this.network.getPriors()[outvar] / this.network.getCountPer()[outvar];
					this.w[sigmaPtr] *= this.network.getPriors()[outvar] / this.network.getCountPer()[outvar];
				}
				this.v[sigmaPtr] /= psum;
				this.w[sigmaPtr] /= psum;
			}

			for (outvar = 0; outvar < this.network.getOutputCount(); outvar++) {
				if (outvar == (int) target.getData(0))
					temp = 2.0 * (out[outvar] - 1.0);
				else
					temp = 2.0 * out[outvar];
				for (int i = 0; i < this.network.getOutputCount(); i++) {
					double vij = v[i * this.network.getInputCount() + ivar];
					double wij = w[i * this.network.getInputCount() + ivar];
					if (i == outvar) {
						der1 = vij * (1.0 - out[outvar]);
						der2 = wij * (1.0 - out[outvar]) + 2.0 * vij * vij * (out[outvar] - 1.0);
					} else {
						der1 = -out[outvar] * vij;
						der2 = out[outvar] * (2.0 * vij * vij - wij);
					}
					this.deriv[i * this.network.getInputCount() + ivar] += temp * der1;
					this.deriv2[i * this.network.getInputCount() + ivar] += temp * der2 + 2.0 * der1 * der1;
				} // For i to nout
			} // For outvar (k in sigma[kij])
		} // For ivar (j in sigma[kij])

		return new BasicMLData(out);
	}
}
