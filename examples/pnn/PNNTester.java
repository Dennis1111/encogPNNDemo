package examples.pnn;

import java.util.Arrays;
import java.util.Random;

import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;

import neural.networks.training.pnn.TrainBasicPNN;
import neural.networks.training.pnn.ga.GeneticAlgoritmBasicPNN;
import neural.pnn.BasicPNN;
import neural.pnn.PNNKernelType;
import neural.pnn.PNNOutputMode;
import neural.pnn.PersistBasicPNN;
import statistics.ConfusionMatrix;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;

import chart.jfree.MyXYDataset;
import chart.jfree.ScatterPlot;
import org.ejml.example.PrincipalComponentAnalysis;

public class PNNTester {

	private Random rnd = new Random();

	// How much noise to add to each input
	// A noisy input should result in a lower prediction value and after
	// training higher sigmas
	private double[] noiseFactor = { 0.2, 0.5 };
	private int classes = 2;
	/*
	 * public PNNTester(int trainSamples, int validationSamples) { MLDataSet
	 * trainDataSet = createDataSet(trainSamples); MLDataSet validationDataSet =
	 * createDataSet(validationSamples); ScatterPlot plot = new
	 * ScatterPlot("Samples", convert(trainDataSet, classes));
	 * plot.showScatter(); PNNKernelType kernel = PNNKernelType.Gaussian;
	 * PNNOutputMode outMode = PNNOutputMode.Classification; int inputCount = 2;
	 * int outputCount = 2; boolean usePriors = true; BasicPNN pnn = new
	 * BasicPNN(kernel, outMode, inputCount, outputCount, false, usePriors);
	 * pnn.setSamples(trainDataSet); TrainBasicPNN train = new
	 * TrainBasicPNN(pnn, trainDataSet); train.setTargetError(0.05); //
	 * train.setNumSigmas(5); train.setSigmaHigh(2.0); train.iteration(1);
	 * pnn.setTrained(true); }
	 */

	public PNNTester(MLDataSet trainDataSet, MLDataSet validationDataSet, int classes, String[] classNames,
			String path) {
		this.classes = classes;
		ScatterPlot plot = new ScatterPlot("Samples", convert(trainDataSet, classNames), "Principal Component 1",
				"Principal Component 2");
		//plot.showScatter();
		PNNKernelType kernel = PNNKernelType.Gaussian;
		PNNOutputMode outMode = PNNOutputMode.Classification;
		int inputCount = trainDataSet.get(0).getInputArray().length;
		int outputCount = classes;
		boolean seperateClasses = false;
		boolean usePriors = false;
		BasicPNN pnn = new BasicPNN(kernel, outMode, inputCount, outputCount, seperateClasses, usePriors);
		pnn.setSamples(trainDataSet);
		double[] sigmas = pnn.getSigma();
		double minSigma = 0.0001;
		double error = pnn.calculateError(trainDataSet);
		System.out.println("TrainError" + error);
		PersistBasicPNN persist = new PersistBasicPNN();
		persist.saveSamples = false;
		/*
		 * try { String filename="D:/pnntest";
		 * System.out.println("Saving"+pnn+Arrays.toString(pnn.getSigma()));
		 * persist.save(new FileOutputStream(filename), pnn); BasicPNN readPNN=
		 * (BasicPNN)persist.read(new FileInputStream(filename));
		 * System.out.println("READ"+readPNN); System.exit(0); } catch
		 * (IOException e) { e.printStackTrace(); }
		 */
		ConfusionMatrix trainCF = pnn.computeConfusionMatrix(trainDataSet);
		System.out.println(trainCF);
		TrainBasicPNN train = new TrainBasicPNN(pnn, trainDataSet);
		train.setValidation(validationDataSet);
		train.setMinSigma(minSigma);
		train.setTargetError(0.05);
		double maxSigma = 2.0;
		train.setSigmaHigh(2.0);
		train.iteration(1);
		pnn.setTrained(true);
		trainCF = pnn.computeConfusionMatrix(trainDataSet);
		System.out.println(trainCF);
		ConfusionMatrix validCF = pnn.computeConfusionMatrix(validationDataSet);
		System.out.println(validCF);

		System.out.println("pnn sigma" + Arrays.toString(pnn.getSigma()));
		// System.exit(0);
		if (seperateClasses)
			path += "Sep";
		int popsize = 200;
		String descr = "Abalone";
		ConfusionMatrix scoreMatrix = new ConfusionMatrix(classNames);
		try {
			GeneticAlgoritmBasicPNN gapnn = new GeneticAlgoritmBasicPNN(pnn, validationDataSet, scoreMatrix, minSigma,
					maxSigma, popsize, path, descr);
			gapnn.evolve(100000);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private MLDataSet createDataSet(int samples) {
		MLDataSet dataSet = new BasicMLDataSet();
		for (int i = 0; i < samples; i++) {
			if (rnd.nextDouble() > 0.5)
				dataSet.add(classOneSample());
			else
				dataSet.add(classTwoSample());
		}
		return dataSet;
	}

	// Class one patterns will be centered at inputs 0,0
	private MLDataPair classOneSample() {
		double[] input = new double[classes];
		input[0] = rnd.nextGaussian() * noiseFactor[0];
		input[1] = input[0] * 0.5 + rnd.nextGaussian() * noiseFactor[1];
		double[] ideal = new double[1];
		ideal[0] = 0;
		MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
		return pair;
	}

	private MLDataPair classTwoSample() {
		double[] input = new double[classes];
		input[0] = 0.5 + rnd.nextGaussian() * noiseFactor[0];
		input[1] = 0.75 - rnd.nextGaussian() * noiseFactor[1];
		double[] ideal = new double[1];
		ideal[0] = 1;
		MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
		return pair;
	}

	private XYDataset convert(MLDataSet dataSet, String[] labels) {
		MyXYDataset xyDataset = new MyXYDataset();
		for (int classNr = 0; classNr < classes; classNr++) {
			XYSeries xySerie = new XYSeries(labels[classNr]);
			xyDataset.addSeries(xySerie);
		}
		double[][] inputs = getInputs(dataSet);
		PrincipalComponentAnalysis pca = getPCA(inputs, 2);
		double[][] reducedInputs = getDimensionReduction(pca, inputs);

		int pattern = 0;
		for (MLDataPair pair : dataSet) {
			// MLData input = pair.getInput();
			int classNr = (int) pair.getIdeal().getData(0);
			xyDataset.addData(reducedInputs[pattern][0], reducedInputs[pattern][1], labels[classNr]);
			pattern++;
		}
		return xyDataset;
	}

	private double[][] getInputs(MLDataSet dataSet) {
		double[][] inputs = new double[dataSet.size()][];
		for (int i = 0; i < dataSet.size(); i++)
			inputs[i] = dataSet.get(i).getInputArray();
		return inputs;
	}

	private PrincipalComponentAnalysis getPCA(double[][] dataSet, int pcaFeatures) {
		int samples = dataSet.length;
		int sampleSize = dataSet[0].length;
		PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();
		pca.setup(samples, sampleSize);
		for (int sample = 0; sample < samples; sample++)
			pca.addSample(dataSet[sample]);
		pca.computeBasis(pcaFeatures);
		return pca;
	}

	private double[][] getDimensionReduction(PrincipalComponentAnalysis pca, double[][] data) {
		int samples = data.length;
		double[][] dimReduction = new double[samples][];
		for (int sample = 0; sample < samples; sample++)
			dimReduction[sample] = pca.sampleToEigenSpace(data[sample]);
		return dimReduction;
	}

	private String getKey(double val) {
		return "Class" + val;
	}

	public static MLDataSet getSubSet(MLDataSet dataSet, int from, int samples) {
		MLDataSet subSet = new BasicMLDataSet();
		for (int i = 0; i < samples; i++) {
			subSet.add(dataSet.get(i));
		}
		return subSet;
	}

	public static void main(String[] args) {
		IrisReader reader=new IrisReader();
		MLDataSet iris = reader.readIris();
		MLDataSet irisTrain = getSubSet(iris, 0, 150);
		MLDataSet irisValid = getSubSet(iris, 100, 150);
		String[] classNames = { "Iris-setosa", "Iris-Versicolor", "Iris-Verginica"};
		String path = "D:/MLDataSet/iris/ga";


	}
}
