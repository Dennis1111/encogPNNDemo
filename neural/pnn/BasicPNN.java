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
package neural.pnn;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.encog.ml.MLClassification;
import org.encog.ml.MLError;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;

import org.encog.neural.NeuralNetworkError;
import org.encog.util.EngineArray;
import org.encog.util.simple.EncogUtility;

import statistics.ConfusionMatrix;

/**
 * This class implements either a:
 * 
 * Probabilistic Neural Network (PNN)
 * 
 * General Regression Neural Network (GRNN)
 * 
 * To use a PNN specify an output mode of classification, to make use of a GRNN
 * specify either an output mode of regression or un-supervised autoassociation.
 * 
 * The PNN/GRNN networks are potentially very useful. They share some
 * similarities with RBF-neural networks and also the Support Vector Machine
 * (SVM). These network types directly support the use of classification.
 * 
 * The following book was very helpful in implementing PNN/GRNN's in Encog.
 * 
 * Advanced Algorithms for Neural Networks: A C++ Sourcebook
 * 
 * by Timothy Masters, PhD (http://www.timothymasters.info/) John Wiley & Sons
 * Inc (Computers); April 3, 1995, ISBN: 0471105880
 */
public class BasicPNN extends AbstractPNN implements MLRegression, MLError, MLClassification {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7990707837655024635L;

	/**
	 * The sigma's specify the widths of each kernel used.
	 */
	private final double[] sigma;

	/**
	 * The training samples that form the memory of this network.
	 */
	private MLDataSet samples;

	/**
	 * Used for classification, the number of cases in each class.
	 */
	private int[] countPer;

	/**
	 * The prior probability weights.
	 */
	private double[] priors;

	private Random rd = new Random();

	// private double maxDistance = 10000;

	/*
	 * The training time grows exponentially with the size of the dataset to
	 * avoid this comparing each sample with a small subset helps and usually
	 * comes close to a full search if the dataset is huge
	 */
	private boolean useSubDataset = false;

	/*
	 * How large subset to compare each sample with;
	 */
	private int subDatasetSize = 200;

	private boolean debug = false;

	private static double[] outSum;

	private String[] classNames;

	/**
	 * Construct a BasicPNN network.
	 * 
	 * @param kernel
	 *            The kernel to use.
	 * @param outmodel
	 *            The output model for this network.
	 * @param inputCount
	 *            The number of inputs in this network.
	 * @param outputCount
	 *            The number of outputs in this network.
	 */

	public BasicPNN(final PNNKernelType kernel, final PNNOutputMode outmodel, final int inputCount,
			final int outputCount, final boolean separateClasses, final boolean usePriors) {
		super(kernel, outmodel, inputCount, outputCount, usePriors);
		classNames = new String[outputCount];
		for (int i = 0; i < outputCount; i++)
			classNames[i] = "Class" + i;
		setSeparateClass(separateClasses);
		if (separateClasses)
			this.sigma = new double[inputCount * outputCount];
		else
			this.sigma = new double[inputCount];
		// To make sure no sigma==0
		java.util.Arrays.fill(sigma, 1.0);
	}

	public String[] getClassNames() {
		return classNames;
	}

	/*
	 *
	 * @param maxCompare how bug subDataset to compare each pattern with
	 */
	public void setSubDataset(int subDatasetSize) {
		this.subDatasetSize = subDatasetSize;
	}

	public void useSubDataset(boolean useSubDataset) {
		this.useSubDataset = useSubDataset;
	}

	public void setSigmas(double[] newSigma) throws Exception {
		if (newSigma.length != sigma.length)
			throw new Exception("sigma length" + sigma.length + "copy sigma" + newSigma.length);
		for (int i = 0; i < sigma.length; i++)
			this.sigma[i] = newSigma[i];
	}

	public int getSigmaLength() {
		return sigma.length;
	}

	public double getSigma(int input, int output) {
		int sigmaPtr = output * this.getInputCount() + input;
		return this.getSigma()[sigmaPtr];
	}

	public void setSigma(int input, int output, double sigmaValue) {
		int sigmaPtr = output * this.getInputCount() + input;
		this.getSigma()[sigmaPtr] = sigmaValue;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	public double[] getLastOutSum() {
		return outSum;
	}

	/**
	 * Compute the output from this network.
	 * 
	 * @param input
	 *            The input to the network.
	 * @return The output from the network.
	 */
	@Override
	public MLData compute(final MLData input) {
		if (this.samples == null)
			System.out.println("Must call setSamples(dataSet) before calling compute");
		final double[] out = new double[getOutputCount()];

		double psum = 0.0;
		int nrOfSamples = this.samples.size();
		int compares = 0;
		MLDataPair pair;
		int sampleIndex = -1;
		double[] patternDensities = new double[nrOfSamples];
		for (final MLDataPair loopPair : this.samples) {
			sampleIndex++;
			if (useSubDataset && compares >= subDatasetSize)
				break;
			if (useSubDataset) {
				int pattern = rd.nextInt(nrOfSamples - 1);
				MLDataPair temp = this.getSamples().get(pattern);
				pair = new BasicMLDataPair(temp.getInput(), temp.getIdeal());
			} else {
				pair = loopPair;
			}

			if (isSamePattern(pair.getInput(), input)) {
				continue;
			}
			double dist = 0.0;
			// When using sep classes scale diff with pairs separate class sigma
			if (isSeparateClass()) {
				int pop = (int) pair.getIdeal().getData(0);
				for (int i = 0; i < getInputCount(); i++) {
					double diff = input.getData(i) - pair.getInput().getData(i);
					int sigmaIndex = pop * getInputCount() + i;
					diff /= this.sigma[sigmaIndex];
					dist += diff * diff;
					/*
					 * if (dist > maxDistance) { sigmaBreakout[sigmaIndex]++;
					 * break; } sigmaDistances[sigmaIndex] += diff * diff;
					 */
				}
			} else
				for (int i = 0; i < getInputCount(); i++) {
					double diff = input.getData(i) - pair.getInput().getData(i);
					diff /= this.sigma[i];
					dist += diff * diff;
					/*
					 * if (dist > maxDistance) { sigmaBreakout[i]++; break; }
					 * sigmaDistances[i] += diff * diff;
					 */
				}
			if (debug && dist > 100000)
				System.out.println("dist" + dist);

			if (getKernel() == PNNKernelType.Gaussian) {
				dist = Math.exp(-dist);
			} else if (getKernel() == PNNKernelType.Reciprocal) {
				dist = 1.0 / (1.0 + dist);
			}

			if (dist < 1.e-40) {
				dist = 1.e-40;
			}

			if (getOutputMode() == PNNOutputMode.Classification) {
				final int pop = (int) pair.getIdeal().getData(0);
				out[pop] += dist;
				// if (debug)
				// System.out.println("pop="+pop+" , out[pop]="+out[pop]);
			} else if (getOutputMode() == PNNOutputMode.Unsupervised) {
				for (int i = 0; i < getInputCount(); i++) {
					out[i] += dist * pair.getInput().getData(i);
				}
				psum += dist;
			} else if (getOutputMode() == PNNOutputMode.Regression) {

				for (int i = 0; i < getOutputCount(); i++) {
					out[i] += dist * pair.getIdeal().getData(i);
				}
				psum += dist;
			}
			patternDensities[sampleIndex] = psum;
			if (psum > 0.1) {
				System.out.println("Very similar pattern");
				System.out.println(pair.getInput().getData());
				System.out.println(input.getData());
			}
			compares++;
		} // end of distance computing (pattern vs pattern)

		outSum = java.util.Arrays.copyOf(out, out.length);
		if (getOutputMode() == PNNOutputMode.Classification) {
			psum = 0.0;
			for (int i = 0; i < getOutputCount(); i++) {
				// Scale up/down out[i] for all classes with their priors so
				// that we expect
				// the sum of out[i] for a dataset to be more equal
				if (usePriors()) {
					out[i] *= ((double) getOutputCount()) / this.priors[i];
				}
				psum += out[i];
			}

			if (psum < 1.e-40) {
				psum = 1.e-40;
			}

			for (int i = 0; i < getOutputCount(); i++) {
				out[i] /= psum;
			}
		} else if (getOutputMode() == PNNOutputMode.Unsupervised) {
			for (int i = 0; i < getInputCount(); i++) {
				out[i] /= psum;
			}
		} else if (getOutputMode() == PNNOutputMode.Regression) {
			for (int i = 0; i < getOutputCount(); i++) {
				out[i] /= psum;
			}
		}

		return new BasicMLData(out);
	}

	/*
	 * private void stats(double[] patternDensities,int patterns) { double
	 * mean=0; for(int i=0;i<patterns;i++) { mean+=patternDensities[i]; }
	 * mean/=patterns; System.out.println(mean); }
	 */

	/**
	 * @return the countPer
	 */
	public int[] getCountPer() {
		return this.countPer;
	}

	/**
	 * @return the priors
	 */
	public double[] getPriors() {
		return this.priors;
	}
	
	public void setPriors(double[] priors)throws Exception{
		if (priors.length !=this.getOutputCount())
			throw new Exception("Priors must have size = "+this.getOutputCount());
		for(int i=0;i<priors.length;i++)
		{
			this.priors[i]=priors[i];
		}
	}

	/**
	 * @return the samples
	 */
	public MLDataSet getSamples() {
		return this.samples;
	}

	/**
	 * @return the sigma
	 */
	public double[] getSigma() {
		return this.sigma;
	}

	/**
	 * @param samples
	 *            the samples to set
	 */
	public void setSamples(final MLDataSet samples) {
		this.samples = samples;

		// update counts per
		if (getOutputMode() == PNNOutputMode.Classification) {

			this.countPer = new int[getOutputCount()];
			this.priors = new double[getOutputCount()];

			for (final MLDataPair pair : samples) {
				final int i = (int) pair.getIdeal().getData(0);
				if (i >= this.countPer.length) {
					throw new NeuralNetworkError(
							"Training data contains more classes than neural network has output neurons to hold.");
				}
				this.countPer[i]++;
			}

			// Let the prior class probability be based on the training set
			for (int i = 0; i < this.priors.length; i++) {
				this.priors[i] = ((double) this.countPer[i]) / samples.size();
				// System.out.println("Prior"+priors[i]);
			}

		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void updateProperties() {
		// unneeded
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double calculateError(MLDataSet data) {
		if (getOutputMode() == PNNOutputMode.Classification) {
			return EncogUtility.calculateClassificationError(this, data);
		} else {
			return EncogUtility.calculateRegressionError(this, data);
		}
	}

	public static boolean isSamePattern(MLData pattern1, MLData pattern2) {
		int length = pattern1.getData().length;
		for (int i = 0; i < length; i++) {
			if (pattern1.getData()[i] != pattern2.getData()[i])
				return false;
		}
		return true;
	}

	public ConfusionMatrix computeConfusionMatrix(MLDataSet dataset) {
		ConfusionMatrix cm = new ConfusionMatrix(classNames);
		for (MLDataPair pair : dataset) {
			final int actual = classify(pair.getInput());
			final int pop = (int) pair.getIdeal().getData(0);
			cm.increase(pop, actual);
		}
		return cm;
	}

	public void setClassNames(String[] classNames) {
		for (int i = 0; i < getOutputCount(); i++)
			this.classNames[i] = classNames[i];
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int classify(MLData input) {
		MLData output = compute(input);
		int classNr = EngineArray.maxIndex(output.getData());
		return classNr;
	}

	@Override
	public List<MLData> compute(MLDataSet dataset) {
		// TODO Auto-generated method stub
		return null;
	}

	public String toString() {
		String description = "BasicPNN\n";
		description += "inputCount= " + this.getInputCount() + "\n";
		description += "outputCount= " + this.getOutputCount() + "\n";
		description += "separateClasses= " + this.isSeparateClass() + "\n";
		description += "usePriors= " + this.usePriors() + "\n";
		description += "Sigmas= " + Arrays.toString(sigma) + "\n";
		return description;
	}

}
