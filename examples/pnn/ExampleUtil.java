package examples.pnn;

import java.util.Random;

import org.ejml.example.PrincipalComponentAnalysis;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;

import chart.jfree.MyXYDataset;

public class ExampleUtil {
	
	private static Random random=new Random();
	
	public static XYDataset convert(MLDataSet dataSet, String[] labels) {
		MyXYDataset xyDataset = new MyXYDataset();
		for (int classNr = 0; classNr < labels.length; classNr++) {
			XYSeries xySerie = new XYSeries(labels[classNr]);
			xyDataset.addSeries(xySerie);
		}
		double[][] inputs = getInputs(dataSet);
		PrincipalComponentAnalysis pca = getPCA(inputs, 2);
		double[][] reducedInputs = getDimensionReduction(pca, inputs);

		int pattern = 0;
		for (MLDataPair pair : dataSet) {
			int classNr = (int) pair.getIdeal().getData(0);
			xyDataset.addData(reducedInputs[pattern][0], reducedInputs[pattern][1], labels[classNr]);
			pattern++;
		}
		return xyDataset;
	}

	private static double[][] getInputs(MLDataSet dataSet) {
		double[][] inputs = new double[dataSet.size()][];
		for (int i = 0; i < dataSet.size(); i++)
			inputs[i] = dataSet.get(i).getInputArray();
		return inputs;
	}

	private static PrincipalComponentAnalysis getPCA(double[][] dataSet, int pcaFeatures) {
		int samples = dataSet.length;
		int sampleSize = dataSet[0].length;
		PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();
		pca.setup(samples, sampleSize);
		for (int sample = 0; sample < samples; sample++)
			pca.addSample(dataSet[sample]);
		pca.computeBasis(pcaFeatures);
		return pca;
	}

	private static double[][] getDimensionReduction(PrincipalComponentAnalysis pca, double[][] data) {
		int samples = data.length;
		double[][] dimReduction = new double[samples][];
		for (int sample = 0; sample < samples; sample++)
			dimReduction[sample] = pca.sampleToEigenSpace(data[sample]);
		return dimReduction;
	}

	public static MLDataSet getSubSet(MLDataSet dataSet, int from, int samples) {
		MLDataSet subSet = new BasicMLDataSet();
		for (int i = 0; i < samples; i++) {
			subSet.add(dataSet.get(i));
		}
		return subSet;
	}
	
	public MLDataSet createArtificialDataSet(int samples) {
		MLDataSet dataSet = new BasicMLDataSet();
		double[] noiseFactor={0.5,1.0};
		for (int i = 0; i < samples; i++) {
			if (random.nextDouble() > 0.3)
				dataSet.add(classOneSample(noiseFactor));
			else
				dataSet.add(classTwoSample(noiseFactor));
		}
		return dataSet;
	}

	// Class one patterns will be centered at inputs 0,0
	private MLDataPair classOneSample(double[] noiseFactor) {
		double[] input = new double[2];
		input[0] = random.nextGaussian() * noiseFactor[0];
		input[1] = input[0] * 0.5 + random.nextGaussian() * noiseFactor[1];
		double[] ideal = new double[1];
		ideal[0] = 0;
		MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
		return pair;
	}

	private MLDataPair classTwoSample(double[] noiseFactor) {
		double[] input = new double[2];
		input[0] = 0.5 + random.nextGaussian() * noiseFactor[0];
		input[1] = 0.75 - random.nextGaussian() * noiseFactor[1];
		double[] ideal = new double[1];
		ideal[0] = 1;
		MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
		return pair;
	}
}
