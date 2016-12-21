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

public class Iris extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 34561L;
	private MLDataSet trainDataSet,validationDataSet;
	private String[] classNames;
	private BasicPNN pnn;
	private JTextArea textArea;
	private String message;
	private JPanel confusionPanel;
	private GridBagConstraints constraint;
	
	public Iris(boolean separateClasses) {
		super("Iris");
		IrisReader reader = new IrisReader();
		this.trainDataSet = reader.readIris();
		this.classNames=reader.getClassNames();
		//We really just interested in the trainDataSet but 
		//the genetic algorititm requires a validationdataset
		validationDataSet = ExampleUtil.getSubSet(trainDataSet, 25, 125);
		this.classNames = reader.getClassNames();
		ScatterPlot plot = new ScatterPlot("Iris Dataset", ExampleUtil.convert(trainDataSet, classNames),
				"Principal Component 1", "Principal Component 2");
		PNNKernelType kernel = PNNKernelType.Gaussian;
		PNNOutputMode outMode = PNNOutputMode.Classification;
		int inputCount = trainDataSet.get(0).getInputArray().length;
		int outputCount = classNames.length;
		boolean usePriors = false;
		this.pnn = new BasicPNN(kernel, outMode, inputCount, outputCount, separateClasses, usePriors);
		pnn.setSamples(trainDataSet);
		pnn.setClassNames(classNames);
		confusionPanel = new JPanel();
		confusionPanel.setLayout(new GridBagLayout());
		constraint = new GridBagConstraints();
		constraint.gridx = 0;
		constraint.gridy = 0;
		constraint.gridwidth = 2;
		textArea = new JTextArea();
		textArea.setEditable(false);
		textArea.setLineWrap(true);
		textArea.setRows(10);
		textArea.setColumns(30);
		message = "Inital Sigmas = " + Arrays.toString(pnn.getSigma()) + "\n";
		textArea.setText(message);
		confusionPanel.add(textArea,constraint);
		constraint.gridwidth = 1;		
		this.add(confusionPanel);
		this.pack();
		this.setVisible(true);
	}

	private double[] getRoundedSigmas() {
		double[] sigma = pnn.getSigma();
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
		String typeOfSearch;
		if (singleSigma)
			typeOfSearch= "One sigma solution";
		else
			typeOfSearch= "Multiple sigma solution";		
		TrainBasicPNN train = new TrainBasicPNN(pnn, trainDataSet);
		train.setSingleSigmaSolution(singleSigma);
		train.setMinSigma(minSigma);
		train.setTargetError(targetError);
		train.setSigmaHigh(maxSigma);
		train.iteration(1);
		pnn.setTrained(true);
		constraint.gridy++;
		ConfusionMatrix trainCM = pnn.computeConfusionMatrix(trainDataSet);
		this.confusionPanel.add(trainCM, constraint);
		trainCM.updateGrid();
		message += "Trained "+typeOfSearch+ "\n"+Arrays.toString(getRoundedSigmas()) + "\n";
		textArea.setText(message);
		this.pack();
	}

	public void geneticTraining(int popSize, double minSigma, double maxSigma) {
		ConfusionMatrix scoreMatrix = new ConfusionMatrix(classNames);
		try {
			String popSavePath="C:/MLDataSet/Iris/GA";
			String description="sep" + pnn.isSeparateClass();
			GeneticAlgoritmBasicPNN gapnn = new GeneticAlgoritmBasicPNN(pnn, validationDataSet, scoreMatrix, minSigma,
					maxSigma, popSize, popSavePath, description);			
			//With default scoreMatrix 1.0 means 100% correct classification
			int generations=1000;
			double solved=1.0;
			gapnn.evolve(generations,solved);
			BasicPNN bestPop = gapnn.getBestIndividual();
			ConfusionMatrix trainCM = bestPop.computeConfusionMatrix(trainDataSet);
			trainCM.setDescription("GeneticAlgoritm Train DataSet");
			trainCM.updateGrid();
			constraint.gridy++;
			constraint.gridx = 0;
			confusionPanel.add(trainCM, constraint);
			message+="Genetic Algoritm Sigmas\n" + Arrays.toString(getRoundedSigmas()) + "\n";
			textArea.setText(message);				
			this.pack();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		Iris iris = new Iris(false);
		// Constrain the Sigma to be >= 0.001
		double minSigma = 0.001;
		double maxSigma = 4;
		// Stop training when trainingError <= targetError
		// Can be set very low as it will stop when 'little' progress is made
		// also
		double targetError = 0.0001;
		iris.basicTraining(true, minSigma, maxSigma, targetError);
		iris.basicTraining(false, minSigma, maxSigma, targetError);
		int popSize=1000;
		iris.geneticTraining(popSize, minSigma,maxSigma);
	}
}
