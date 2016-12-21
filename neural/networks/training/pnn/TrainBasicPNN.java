/*
 * Encog(tm) Core v3.2 - Java Version
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-core
 
 * Copyright 2008-2013 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package neural.networks.training.pnn;

import java.util.concurrent.Future;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import org.encog.ml.MLMethod;
import org.encog.ml.TrainingImplementationType;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.training.propagation.TrainingContinuation;
import org.encog.util.EngineArray;

import neural.pnn.BasicPNN;
import neural.pnn.PNNKernelType;
import neural.pnn.PNNOutputMode;

/**
 * Train a PNN.
 */
public class TrainBasicPNN extends BasicTraining implements CalculationCriteria, CalcError {

	/**
	 * The default target error.
	 */
	public static final double DEFAULT_TARGET_ERROR = 0.05;

	/**
	 * The default minimum improvement before stop.
	 */
	public static final double DEFAULT_MIN_IMPROVEMENT = 0.0001;

	/**
	 * THe default sigma low value.
	 */
	public static final double DEFAULT_GLOB_MIN_SEARCH_LOW = 0.1;

	/**
	 * The default sigma high value.
	 */
	public static final double DEFAULT_GLOB_MIN_SEARCH_HIGH = 10.0;

	/**
	 * The default number of sigmas to evaluate between the low and high.)
	 */
	public static final int DEFAULT_GLOB_MIN_SEARCH_POINTS = 10;

	/**
	 * Temp storage for derivative computation.
	 */
	private double[] v;

	/**
	 * Temp storage for derivative computation.
	 */
	private double[] w;

	/**
	 * Temp storage for derivative computation.
	 */
	private double[] dsqr;

	/**
	 * The network to train.
	 */
	private final BasicPNN network;

	/**
	 * The training data.
	 */
	private final MLDataSet training;

	/**
	 * Stop training when error goes below targetError.
	 */
	private double targetError;

	/**
	 * The minimum improvement allowed.
	 */
	private double minImprovement;

	/**
	 * The low value for the sigma search.
	 */
	private double globMinSearchLow;

	/**
	 * The high value for the sigma search.
	 */
	private double globMinSearchHigh = 10.0;

	/**
	 * The number of sigmas to evaluate between the low and high.
	 */
	private int numSearchPoints = DEFAULT_GLOB_MIN_SEARCH_POINTS;

	/**
	 * Have the samples been loaded.
	 */
	private boolean samplesLoaded;

	private double minSigma = 0.0001;

	private TrainBasicPNNWorker[] workers;

	protected MLDataSet validation = null;

	private int threads = 4;

	private boolean singleSigmaSolution = false;

	/**
	 * Train a BasicPNN.
	 * 
	 * @param network
	 *            The network to train.
	 * @param training
	 *            The training data.
	 */
	public TrainBasicPNN(final BasicPNN network, final MLDataSet training) {
		super(TrainingImplementationType.OnePass);
		this.network = network;
		this.training = training;
		this.targetError = TrainBasicPNN.DEFAULT_TARGET_ERROR;
		this.minImprovement = TrainBasicPNN.DEFAULT_MIN_IMPROVEMENT;
		this.globMinSearchLow = TrainBasicPNN.DEFAULT_GLOB_MIN_SEARCH_LOW;
		this.globMinSearchHigh = TrainBasicPNN.DEFAULT_GLOB_MIN_SEARCH_HIGH;
		// this.numSigmas = TrainBasicPNN.DEFAULT_NUM_SIGMAS;
		this.samplesLoaded = false;
	}

	public void setThreads(int threads) {
		this.threads = threads;
	}

	public void setSingleSigmaSolution(boolean singleSigmaSolution) {
		this.singleSigmaSolution = singleSigmaSolution;
	}

	public void setValidation(MLDataSet dataSet) {
		this.validation = dataSet;
	}

	public void setMinSigma(double minSigma) {
		this.minSigma = minSigma;
	}

	private boolean isSigmaOK(double[] sigma) {
		for (double val : sigma)
			if (val < minSigma)
				return false;
		return true;
	}

	/**
	 * Calculate the error with multiple sigmas.
	 * 
	 * @param x
	 *            The data.
	 * @param der1
	 *            The first derivative.
	 * @param der2
	 *            The 2nd derivatives.
	 * @param der
	 *            Calculate the derivative.
	 * @return The error.
	 */
	@Override
	public double calcErrorWithMultipleSigma(final double[] x, final double[] der1, final double[] der2,
			final boolean useDeriv) {
		if (!isSigmaOK(x)) {
			System.out.println("BAD SIGMA" + Arrays.toString(x));
			System.exit(0);
		}

		setNetworkSigmas(x);

		double error;
		if (!useDeriv) {
			return calculateErrorWithWorkers(this.network.getSamples(), useDeriv, x, der1, der2);
		}

		error = calculateErrorWithWorkers(training, useDeriv, x, der1, der2);
		return error;

	}

	/**
	 * Calculate the error using a common sigma. Will have the side effect of
	 * setting the networks sigma
	 * 
	 * @param sig
	 *            The sigma to use.
	 * @return The training error.
	 */
	@Override
	public double calcErrorWithSingleSigma(final double sig) {
		double[] sigmaCopy = EngineArray.arrayCopy(this.network.getSigma());
		int ivar;
		for (ivar = 0; ivar < this.network.getSigma().length; ivar++) {
			this.network.getSigma()[ivar] = sig;
		}

		double errorWorkers = calculateErrorWithWorkers(this.network.getSamples(), false, sigmaCopy, null, null);
		return errorWorkers;
	}

	private void setNetworkSigmas(double[] source) {
		for (int ivar = 0; ivar < this.network.getInputCount(); ivar++) {
			this.network.getSigma()[ivar] = source[ivar];
		}
	}

	public double calculateErrorWithWorkers(final MLDataSet training, final boolean useDeriv, final double[] sigma,
			final double[] deriv, final double[] deriv2) {
		workers = new TrainBasicPNNWorker[threads];
		int patterns = (int) training.getRecordCount();
		double workerPatterns = ((double) patterns) / workers.length;
		// Almost Infinity Don't wanna time out on large calculations
		int keepAliveTime = 100000000;
		TimeUnit timeUnit = TimeUnit.SECONDS;
		BlockingQueue<Runnable> workQueue = new LinkedBlockingQueue<Runnable>();
		ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(threads, threads, keepAliveTime, timeUnit,
				workQueue);
		ExecutorCompletionService<TrainBasicPNNWorkerResult> ecs = new ExecutorCompletionService<TrainBasicPNNWorkerResult>(
				threadPoolExecutor);
		List<Future<TrainBasicPNNWorkerResult>> futures = new ArrayList<Future<TrainBasicPNNWorkerResult>>();

		for (int workerCount = 0; workerCount < workers.length; workerCount++) {
			int from = (int) workerPatterns * workerCount;
			int to = ((int) workerPatterns * (workerCount + 1)) - 1;
			if (workerCount == workers.length - 1)
				to = patterns - 1;
			workers[workerCount] = new TrainBasicPNNWorker(this.network, training, useDeriv, deriv, deriv2, sigma, from,
					to);
			Future<TrainBasicPNNWorkerResult> future = ecs.submit(workers[workerCount]);
			futures.add(future);
		}

		double averageError = 0;
		double[] workerDeriv = null, workerDeriv2 = null;
		if (useDeriv) {
			workerDeriv = new double[sigma.length];
			java.util.Arrays.fill(workerDeriv, 0.0);
			workerDeriv2 = new double[sigma.length];
			java.util.Arrays.fill(workerDeriv2, 0.0);
		}

		// Get the results when tasks gets finished
		for (int result = 0; result < workers.length; result++) {
			try {
				Future<TrainBasicPNNWorkerResult> future = ecs.take();
				TrainBasicPNNWorkerResult workerResult = future.get();
				averageError += workerResult.getError();
				if (useDeriv) {
					for (int i = 0; i < deriv.length; i++) {
						workerDeriv[i] += workerResult.getDeriv()[i];
						workerDeriv2[i] += workerResult.getDeriv2()[i];
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
		if (useDeriv) {
			for (int i = 0; i < deriv.length; i++) {
				deriv[i] = workerDeriv[i] / patterns;
				deriv2[i] = workerDeriv2[i] / patterns;
			}
		}
		averageError /= patterns;
		return averageError;
	}

	/**
	 * Calculate the error for the entire training set.
	 * 
	 * @param training
	 *            Training set to use.
	 * @param deriv
	 *            Should we find the derivative.
	 * @return The error.
	 */

	public double calculateError(final MLDataSet training, final boolean deriv) {

		double err, totErr;
		double diff;
		totErr = 0.0;

		if (deriv) {
			final int num = (this.network.isSeparateClass())
					? this.network.getInputCount() * this.network.getOutputCount() : this.network.getInputCount();
			for (int i = 0; i < num; i++) {
				this.network.getDeriv()[i] = 0.0;
				this.network.getDeriv2()[i] = 0.0;
			}
		}

		// this.network.setExclude((int) training.getRecordCount());

		final MLDataPair pair = BasicMLDataPair.createPair(training.getInputSize(), training.getIdealSize());

		final double[] out = new double[this.network.getOutputCount()];
		System.out.print("TrainBasicPNN RC" + training.getRecordCount());
		for (int r = 0; r < training.getRecordCount(); r++) {
			if (r % 100 == 0)
				System.out.print(r + ",");
			training.getRecord(r, pair);

			err = 0.0;

			final MLData input = pair.getInput();
			final MLData target = pair.getIdeal();

			if (this.network.getOutputMode() == PNNOutputMode.Unsupervised) {
				if (deriv) {
					final MLData output = computeDeriv(input, target);
					for (int z = 0; z < this.network.getOutputCount(); z++) {
						out[z] = output.getData(z);
					}
				} else {
					final MLData output = this.network.compute(input);
					for (int z = 0; z < this.network.getOutputCount(); z++) {
						out[z] = output.getData(z);
					}
				}
				for (int i = 0; i < this.network.getOutputCount(); i++) {
					diff = input.getData(i) - out[i];
					err += diff * diff;
				}
			} else if (this.network.getOutputMode() == PNNOutputMode.Classification) {
				final int tclass = (int) target.getData(0);
				MLData output;

				if (deriv) {
					output = computeDeriv(input, pair.getIdeal());
				} else {
					output = this.network.compute(input);
				}

				EngineArray.arrayCopy(output.getData(), out);

				for (int i = 0; i < out.length; i++) {
					if (i == tclass) {
						diff = 1.0 - out[i];
						err += diff * diff;
					} else {
						err += out[i] * out[i];
					}
				}
			} // End classification

			else if (this.network.getOutputMode() == PNNOutputMode.Regression) {
				if (deriv) {
					final MLData output = this.network.compute(input);
					for (int z = 0; z < this.network.getOutputCount(); z++) {
						out[z] = output.getData(z);
					}
				} else {
					final MLData output = this.network.compute(input);
					for (int z = 0; z < this.network.getOutputCount(); z++) {
						out[z] = output.getData(z);
					}
				}
				for (int i = 0; i < this.network.getOutputCount(); i++) {
					diff = target.getData(i) - out[i];
					err += diff * diff;
				}
			}
			totErr += err;
		}

		this.network.setError(totErr / training.getRecordCount());
		if (deriv) {
			for (int i = 0; i < this.network.getDeriv().length; i++) {
				this.network.getDeriv()[i] /= training.getRecordCount();
				this.network.getDeriv2()[i] /= training.getRecordCount();
			}
		}

		if ((this.network.getOutputMode() == PNNOutputMode.Unsupervised)
				|| (this.network.getOutputMode() == PNNOutputMode.Regression)) {
			this.network.setError(this.network.getError() / this.network.getOutputCount());
			if (deriv) {
				for (int i = 0; i < this.network.getInputCount(); i++) {
					this.network.getDeriv()[i] /= this.network.getOutputCount();
					this.network.getDeriv2()[i] /= this.network.getOutputCount();
				}
			}
		}
		return this.network.getError();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public boolean canContinue() {
		return false;
	}

	/**
	 * Compute the derivative for target data.
	 * 
	 * @param input
	 *            The input.
	 * @param target
	 *            The target data.
	 * @return The output.
	 */
	public MLData computeDeriv(final MLData input, final MLData target) {
		int pop, ivar;
		int outvar;
		double diff, dist, truedist;
		double vtot, wtot;
		double temp, der1, der2, psum;
		int vptr, wptr, vsptr = 0, wsptr = 0;

		final double[] out = new double[this.network.getOutputCount()];

		for (pop = 0; pop < this.network.getOutputCount(); pop++) {
			out[pop] = 0.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				this.v[pop * this.network.getInputCount() + ivar] = 0.0;
				this.w[pop * this.network.getInputCount() + ivar] = 0.0;
			}
		}

		psum = 0.0;

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
		for (int r = 0; r < nrOfSamples; r++) {
			this.network.getSamples().getRecord(r, pair);
			// int exclude = nrOfSamples - 1 - this.network.getExclude();
			// if (r == exclude)
			if (BasicPNN.isSamePattern(pair.getInput(), input)) {
				continue;
			}

			dist = 0.0;
			for (ivar = 0; ivar < this.network.getInputCount(); ivar++) {
				diff = input.getData(ivar) - pair.getInput().getData(ivar);
				diff /= this.network.getSigma()[ivar];
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
			}
		}

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
				this.network.getDeriv()[ivar] += temp * der1;
				this.network.getDeriv2()[ivar] += temp * der2 + 2.0 * der1 * der1;
			}
		}
		return new BasicMLData(out);
	}

	/**
	 * @return the maxError
	 */
	public double getTargetError() {
		return this.targetError;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public MLMethod getMethod() {
		return this.network;
	}

	/**
	 * @return the minImprovement
	 */
	public double getMinImprovement() {
		return this.minImprovement;
	}

	/**
	 * @return the numSigmas
	 */
	/*
	 * public int getNumSigmas() { return this.numSigmas; }
	 */

	/**
	 * @return the sigmaHigh
	 */
	public double getSigmaHigh() {
		return this.globMinSearchHigh;
	}

	/**
	 * @return the sigmaLow
	 */
	public double getSigmaLow() {
		return this.globMinSearchLow;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void iteration() {
		preIteration();

		if (!this.samplesLoaded) {
			this.network.setSamples(new BasicMLDataSet(this.training));
			this.samplesLoaded = true;
		}

		final GlobalMinimumSearch globalMinimum = new GlobalMinimumSearch();
		final DeriveMinimum deriveMin = new DeriveMinimum();

		int sigmaLength = this.network.getSigma().length;

		this.dsqr = new double[this.network.getInputCount()];
		this.v = new double[sigmaLength];
		this.w = new double[sigmaLength];
		final double[] x = new double[sigmaLength];
		for (int i = 0; i < sigmaLength; i++) {
			x[i] = this.network.getSigma()[i];
		}
		if (this.singleSigmaSolution) {
			boolean firstPointKnown = false;
			globalMinimum.findBestRange(this.minSigma, this.globMinSearchHigh, this.numSearchPoints, false, targetError,
					this, firstPointKnown);
			firstPointKnown = true;
			for (int iteration = 0; iteration < 5; iteration++) {
				globalMinimum.findBestRange(globalMinimum.getX1(), globalMinimum.getX3(), this.numSearchPoints, false,
						targetError, this, firstPointKnown);
				globalMinimum.brentmin(10, targetError, 1.e-6, 1.e-5, this, globalMinimum.getY2());
			}
		} else {
			int maxIterations = 32767;
			maxIterations = 1000;
			double eps = 1.e-8;// Machine precision!?
			final double d = deriveMin.calculate(maxIterations, minSigma, this.targetError, eps, this.minImprovement,
					this, sigmaLength, x, globalMinimum.getY2());
			globalMinimum.setY2(d);

			for (int i = 0; i < sigmaLength; i++) {
				this.network.getSigma()[i] = x[i];
			}
		}
		this.network.setError(Math.abs(globalMinimum.getY2()));
		this.network.setTrained(true); // Tell other routines net is trained
		this.setError(network.getError());
		postIteration();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public TrainingContinuation pause() {
		return null;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void resume(final TrainingContinuation state) {
	}

	/**
	 * @param targetError
	 *            the targetError to set
	 */
	public void setTargetError(final double targetError) {
		this.targetError = targetError;
	}

	/**
	 * @param minImprovement
	 *            the minImprovement to set
	 */
	public void setMinImprovement(final double minImprovement) {
		this.minImprovement = minImprovement;
	}

	/**
	 * @param numSigmas
	 *            the numSigmas to set
	 */

	public void setNumSigmas(final int numSigmas) {
		this.numSearchPoints = numSigmas;
	}

	/**
	 * @param sigmaHigh
	 *            the sigmaHigh to set
	 */
	public void setSigmaHigh(final double sigmaHigh) {
		this.globMinSearchHigh = sigmaHigh;
	}

	/**
	 * @param sigmaLow
	 *            the sigmaLow to set
	 */
	public void setSigmaLow(final double sigmaLow) {
		this.globMinSearchLow = sigmaLow;
	}

	public double calcError(double sigma) {
		if (sigma <= 0)
			return 1000000;
		double error = calcErrorWithSingleSigma(sigma);
		return error;
	}
}
