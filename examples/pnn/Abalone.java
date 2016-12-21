package examples.pnn;


import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.util.Arrays;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextArea;

import org.encog.ml.data.MLDataSet;

import neural.networks.training.pnn.TrainBasicPNN;
import neural.networks.training.pnn.ga.GeneticAlgoritmBasicPNN;
import neural.pnn.BasicPNN;
import neural.pnn.PNNKernelType;
import neural.pnn.PNNOutputMode;
import statistics.ConfusionMatrix;

import chart.jfree.ScatterPlot;

public class Abalone extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 34561L;
	private MLDataSet trainDataSet, validationDataSet;
	private String[] classNames = { "Age 0-8", "Age 9-10", "Age 11.." };
	private BasicPNN pnn;
	private JTextArea textArea;
	private String message;
	private JPanel confusionPanel;
	private GridBagConstraints constraint;

	public Abalone(boolean separateClasses) {
		super("Abalone");
		MLDataSet abalone = AbaloneReader.readAbalone();
		// Split the dataset as in abalone project
		// The first 3133 examples will be trainingSet
		trainDataSet = ExampleUtil.getSubSet(abalone, 0, 3133);
		// The last 1044 examples will be validationSet
		validationDataSet = ExampleUtil.getSubSet(abalone, 3133, 1044);
		ScatterPlot plot = new ScatterPlot("Samples", ExampleUtil.convert(trainDataSet, classNames),
				"Principal Component 1", "Principal Component 2");
		PNNKernelType kernel = PNNKernelType.Gaussian;
		PNNOutputMode outMode = PNNOutputMode.Classification;
		int inputCount = trainDataSet.get(0).getInputArray().length;
		int outputCount = classNames.length;
		boolean usePriors = false;
		this.pnn = new BasicPNN(kernel, outMode, inputCount, outputCount, separateClasses, usePriors);
		pnn.setSamples(trainDataSet);
		pnn.setClassNames(classNames);
		ConfusionMatrix trainCF = pnn.computeConfusionMatrix(trainDataSet);
		trainCF.setDescription("Train DataSet Init");
		trainCF.updateGrid();
		ConfusionMatrix validCF = pnn.computeConfusionMatrix(validationDataSet);

		validCF.setDescription("Valid DataSet Init");
		validCF.updateGrid();

		textArea = new JTextArea();
		textArea.setEditable(false);
		textArea.setLineWrap(true);
		textArea.setRows(10);
		textArea.setColumns(30);
		confusionPanel = new JPanel();
		confusionPanel.setLayout(new GridBagLayout());
		constraint = new GridBagConstraints();
		constraint.gridx = 0;
		constraint.gridy = 0;
		constraint.gridwidth = 2;
		confusionPanel.add(textArea,constraint);
		constraint.gridwidth = 1;
		message = "Starting Sigmas\n" + Arrays.toString(pnn.getSigma()) + "\n";
		textArea.setText(message);
		this.add(confusionPanel);
		this.pack();
		this.setVisible(true);
	}

	private double[] getRoundedSigmas(double[] sigma) {
		double[] rounded = new double[sigma.length];
		for (int i = 0; i < sigma.length; i++)
			rounded[i] = Math.rint(sigma[i] * 1000) / 1000;
		return rounded;
	}

	/*
	 * @singleSigma when true the same sigmaValue has to be applied to all
	 * inputs
	 * 
	 * @minSigma Constrain the training to generate sigmas >=minSigma
	 * 
	 * @maxSigma Constrain the training to generate sigmas <=maxSigma
	 * 
	 * @targetError When the traininingError reaches this error we stop the
	 * iterations
	 */
	public void basicTraining(boolean singleSigma, double minSigma, double maxSigma, double targetError) {
		TrainBasicPNN train = new TrainBasicPNN(pnn, trainDataSet);
		train.setSingleSigmaSolution(singleSigma);
		train.setValidation(validationDataSet);
		train.setMinSigma(minSigma);
		train.setTargetError(targetError);
		train.setSigmaHigh(maxSigma);
		train.setMinImprovement(0.00001);
		train.iteration(1);
		pnn.setTrained(true);
		String sigmaOption;
		if (singleSigma)
			sigmaOption = "One Sigma" + "\n";
		else
			sigmaOption = "Multiple Sigma" + "\n";
		ConfusionMatrix trainCM = pnn.computeConfusionMatrix(trainDataSet);
		trainCM.setDescription("Train DataSet with " + sigmaOption);
		trainCM.updateGrid();
		constraint.gridy++;
		constraint.gridx = 0;
		this.confusionPanel.add(trainCM, constraint);
		ConfusionMatrix validCM = pnn.computeConfusionMatrix(validationDataSet);
		validCM.setDescription("Validation Data Set with " + sigmaOption);
		validCM.updateGrid();
		constraint.gridx = 1;
		this.confusionPanel.add(validCM, constraint);
		message += "Trained " + sigmaOption + Arrays.toString(getRoundedSigmas(pnn.getSigma())) + "\n";
		textArea.setText(message);
		this.pack();
	}

	public void geneticTraining(int popSize, double minSigma, double maxSigma) {
		ConfusionMatrix cm = new ConfusionMatrix(classNames);
		//It's possible to change learning target as an example
		//cm.setScoreMatrix(-1, 2, 0); 
		//would punish Age11.. classified as Age0-8 harder during training
		try {
			String popSavePath="C:/MLDataSet/Abalone/GA";
			String description="sep" + pnn.isSeparateClass();
			GeneticAlgoritmBasicPNN gapnn = new GeneticAlgoritmBasicPNN(pnn, validationDataSet, cm, minSigma,
					maxSigma, popSize,popSavePath, description);
			int generations=50;
			double solved=1.0;
			gapnn.evolve(generations,solved);
			BasicPNN bestPop = gapnn.getBestIndividual();
			ConfusionMatrix trainCM = bestPop.computeConfusionMatrix(trainDataSet);
			trainCM.setDescription("GeneticAlgoritm Train DataSet");
			trainCM.updateGrid();
			constraint.gridy++;
			constraint.gridx = 0;
			this.confusionPanel.add(trainCM, constraint);
			message += "Trained Sigma After Genetic Algoritm \n" + Arrays.toString(getRoundedSigmas(bestPop.getSigma()))
					+ "\n";
			textArea.setText(message);
			ConfusionMatrix validCM = bestPop.computeConfusionMatrix(validationDataSet);
			validCM.setDescription("GeneticAlgoritm Validation Data Set");
			validCM.updateGrid();
			constraint.gridx = 1;
			this.confusionPanel.add(validCM, constraint);
			this.pack();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		Abalone abalone = new Abalone(false);
		// Constrain the Sigma to be >= 0.001
		double minSigma = 0.001;
		double maxSigma = 4;
		// Stop training when trainingError <= targetError
		// Can be set very low as it will stop when 'little' progress is made
		// also
		double targetError = 0.1;
		boolean singleSigmaSolution = true;
		abalone.basicTraining(singleSigmaSolution, minSigma, maxSigma, targetError);
		abalone.basicTraining(false, minSigma, maxSigma, targetError);
		int popSize = 1000;
		abalone.geneticTraining(popSize, minSigma, maxSigma);
	}
}
