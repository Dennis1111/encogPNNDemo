package neural.networks.training.pnn.ga;

/* 
 * This program has used tiny_gp as skeleton written by
 * Riccardo Poli (email: rpoli@essex.ac.uk).
 * http://cswww.essex.ac.uk/staff/rpoli/TinyGP/    
 * 
 * @Author Dennis Nilsson
 */

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.encog.ml.data.MLDataSet;

import chart.jfree.LineChart;
import javafx.util.Pair;
import neural.pnn.BasicPNN;
import neural.pnn.PersistBasicPNN;
import statistics.ConfusionMatrix;
import util.MyLogger;

public class GeneticAlgoritmBasicPNN implements ActionListener {

	private PersistBasicPNN persistPNN;
	private MyLogger log = null;
	private MLDataSet validDataSet;
	private ConfusionMatrix scoreMatrix;

	private double[] fitness;
	private BasicPNN[] population;
	private Map<BasicPNN, ConfusionMatrix> pnnCfMatrixTrain;
	private Map<BasicPNN, ConfusionMatrix> pnnCfMatrixValid;
	private final double MIN_SIGMA;
	private final double MAX_SIGMA;
	private double temperature = 1.0;
	static Random rd = new Random();

	private final int POPSIZE;
	private final int GENERATIONS = 100;
	private final int TOURNAMENT_SIZE = 25;
	private final double PMUT_PER_NODE = 0.01;
	private final double CROSSOVER_PROB = 0.01;
	private final double MUTATION_PROB = 0.10;
	private int generation = 0;
	private String path;
	private String description;

	private LineChart lineChart;
	private final static String TRAIN_AVERAGE = "Train Average";
	private final static String TRAIN_BEST = "Train Best";
	private final static String TRAIN_WORST = "Train Worst";
	private final static String VALID_WORST = "Eval Worst";
	private final static String VALID_BEST = "Eval Best";
	private boolean stopEvolution = false;
	private JButton stopEvolutionButton = null;

	public GeneticAlgoritmBasicPNN(BasicPNN pnn, MLDataSet validDataSet, ConfusionMatrix scoreMatrix, double minSigma,
			double maxSigma, int popsize, String path, String descr) throws Exception {
		if (pnn.getSamples() == null)
			throw new Exception("PNN requires a training set for GA");
		System.out.println(path + " : " + descr);
		try {
			// Create the path if it doesnt exist
			new File(path).mkdirs();
			this.log = new MyLogger(path + "/" + GeneticAlgoritmBasicPNN.class.getName() + ".log");
			this.log.println("log init" + new java.util.Date().toString());
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
		this.scoreMatrix = scoreMatrix;
		this.validDataSet = validDataSet;
		this.MIN_SIGMA = minSigma;
		this.MAX_SIGMA = maxSigma;
		this.POPSIZE = popsize;
		this.fitness = new double[POPSIZE];
		pnnCfMatrixTrain = new HashMap<>();
		pnnCfMatrixValid = new HashMap<>();
		this.path = path;
		this.description = descr;
		// Create a directory if doesnt exist yet
		// new File(path).mkdirs();
		persistPNN = new PersistBasicPNN();
		persistPNN.saveSamples = false;
		try {
			loadPopulation(path, descr, pnn, pnn.getSamples());
		} catch (Exception e) {
			System.out.println("Couldn't load population" + e);
			e.printStackTrace();
			population = createRandomPop(pnn, POPSIZE, fitness);
			try {
				savePopulation(path, description);
			} catch (IOException ioe) {
				ioe.printStackTrace();
				System.exit(0);
			}
		}
	}

	private JFrame initChart() {
		this.lineChart = new LineChart("PNN Genetic Evolution", "Generation", "Population Scores");
		JFrame frame = new JFrame();
		frame.setSize(1400, 1200);
		frame.getContentPane().add(lineChart);
		JPanel buttonPanel = new JPanel();
		stopEvolutionButton = new JButton("Stop Evolution");
		buttonPanel.add(stopEvolutionButton);	
		stopEvolutionButton.addActionListener(this);
		frame.getContentPane().add(buttonPanel, BorderLayout.SOUTH);
		frame.pack();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		return frame;
	}

	/**
	 * Invoked when an action occurs.
	 *
	 * @param e
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		System.out.println("Action event");
		if (e.getSource() == this.stopEvolutionButton) {
			System.out.println("STOP");
			this.stopEvolution = true;
		}
	}

	private double fitnessFunction(BasicPNN pnn, MLDataSet dataset, Map<BasicPNN, ConfusionMatrix> pnnMatrixMap) {
		ConfusionMatrix cm = pnn.computeConfusionMatrix(dataset);
		double score = cm.getScore(scoreMatrix) / cm.getNrOfPatterns();
		pnnMatrixMap.put(pnn, cm);
		//System.out.println("New fit score" + score + "patt" + cm.getNrOfPatterns());
		return score;
	}

	private BasicPNN createRandomIndiv(BasicPNN pnn) {
		BasicPNN individual = clonePNN(pnn);
		individual.setSamples(pnn.getSamples());
		double[] sigmas = individual.getSigma();
		double randomRange = (MAX_SIGMA - MIN_SIGMA) * 0.1;
		for (int sigma = 0; sigma < sigmas.length; sigma++) {
			double rndSigma = rd.nextDouble() * randomRange;
			sigmas[sigma] = Math.min(rndSigma, MAX_SIGMA);
			sigmas[sigma] = Math.max(rndSigma, MIN_SIGMA);
		}
		return individual;
	}

	private BasicPNN[] createRandomPop(BasicPNN pnn, int n, double[] fitness) {
		ExecutorService pool = Executors.newFixedThreadPool(4);
		List<Future<Pair<BasicPNN, Double>>> futures = new ArrayList<>();
		BasicPNN[] pop = new BasicPNN[n];
		for (int i = 0; i < n; i++) {
			pop[i] = createRandomIndiv(pnn);
			System.out.println(pop[i]);
			MLDataSet dataset = pnn.getSamples();
			futures.add(pool.submit(new GeneticWorker(pop[i], dataset, pnnCfMatrixTrain)));
		}
		int i = 0;
		for (Future<Pair<BasicPNN, Double>> future : futures) {
			try {
				BasicPNN indiv = future.get().getKey();
				double indivFitness = future.get().getValue().doubleValue();
				pop[i] = indiv;
				fitness[i] = indivFitness;
				persistPNN.save(getOutputStream(i), pop[i]);
				System.out.println("created random and saved indiv" + i + "fitness" + fitness[i]);
				i++;
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
		pool.shutdown();
		return pop;
	}

	// Return the elitePerc as an list of Integer to be used as indexes in
	// population[]
	private List<Integer> getElitePop(double elitePerc) {
		List<Integer> elitePop = new ArrayList<>();
		int eliteSize = (int) (POPSIZE * elitePerc);
		while (elitePop.size() < eliteSize) {
			// Find best remaining indiv
			double bestScore = 0;// We maximize scores
			int bestIndiv = 0;
			for (int indiv = 0; indiv < POPSIZE; indiv++) {
				if (!elitePop.contains(indiv) && fitness[indiv] > bestScore) {
					bestIndiv = indiv;
					bestScore = fitness[indiv];
				}
			}
			elitePop.add(bestIndiv);
		}
		return elitePop;
	}

	private double populationScore(List<Integer> subset) {
		double popDiversity = popDiversity(subset);
		double averageScore = 0;
		for (Integer indiv : subset) {
			averageScore += fitness[indiv];
		}
		averageScore /= subset.size();
		double popScore = averageScore + popDiversity;
		//System.out.println("pop" + popScore + "Average" + averageScore + "popVar" + popDiversity);
		return popScore;
	}

	// Treat two individuals as different if one of their sigma>minDiff and one
	// of those sigma<maxSigma (diff between large sigmas doesn't mean so much)
	// Then count how many unique people we have ie one unique person has no
	// similar person

	private double popDiversity(List<Integer> pop) {
		double minDiff = 0.2;
		double maxSigma = 1.0;
		List<BasicPNN> uniqueIndivuals = new ArrayList<>();
		for (Integer cand : pop) {
			boolean candIsUnique = true;

			for (BasicPNN uniqueIndividual : uniqueIndivuals)
				if (!different(population[cand].getSigma(), uniqueIndividual.getSigma(), minDiff, maxSigma)) {
					candIsUnique = false;
					// System.out.println("Uniq"+Arrays.toString(uniqueIndividual.getSigma()));
					break;
				}
			if (candIsUnique)
				uniqueIndivuals.add(population[cand]);
		}
		double diversity = ((double) uniqueIndivuals.size()) / pop.size();
		return diversity;
	}

	private boolean different(double[] sig1, double[] sig2, double minDiff, double maxSigma) {
		for (int i = 0; i < sig1.length; i++) {
			double diff = sig1[i] - sig2[i];
			double smallestSigma = Math.min(sig1[i], sig2[i]);
			if (diff > minDiff && smallestSigma < maxSigma)
				return true;
		}
		return false;
	}

	private InputStream getInputStream(int population) throws IOException {
		return new FileInputStream(path + "/" + description + population);
	}

	private OutputStream getOutputStream(int population) throws IOException {
		return getOutputStream(path + "/" + description + population);
	}

	private OutputStream getOutputStream(String file) throws IOException {
		OutputStream os = new FileOutputStream(file);
		return os;
	}

	private StatsRecord stats(double[] fitness, BasicPNN[] pop, int gen) {
		StatsRecord stats = new StatsRecord();
		log.println("Statistics POP");
		stats.best = 0;
		stats.worst = 0;
		stats.bestValue = fitness[stats.best];
		stats.worstValue = fitness[stats.worst];
		stats.averageValue = 0.0;

		for (int i = 0; i < POPSIZE; i++) {
			stats.averageValue += fitness[i];
			if (fitness[i] > stats.bestValue) {
				stats.best = i;
				stats.bestValue = fitness[i];
			}
			if (fitness[i] < stats.worstValue) {
				stats.worst = i;
				stats.worstValue = fitness[i];
			}
		}
		stats.averageValue /= POPSIZE;
		log.println("Generation=" + gen + " Avg Fitness=" + (stats.averageValue) + "\nWorst fitness" + stats.worstValue
				+ "Best Fitness=" + (stats.bestValue) + "\nBest Individual" + stats.best + ": ");
		if (pnnCfMatrixTrain.get(pop[stats.best]) == null)
			fitnessFunction(population[stats.best], population[stats.best].getSamples(), pnnCfMatrixTrain);
		System.out.println("getCMTrain " + pnnCfMatrixTrain.get(pop[stats.best]));
		log.println("getCMTrain " + pnnCfMatrixTrain.get(pop[stats.best]));
		if (pnnCfMatrixValid.get(pop[stats.best]) == null)
			fitnessFunction(population[stats.best], validDataSet, pnnCfMatrixValid);
		log.println("getCMValid " + pnnCfMatrixValid.get(pop[stats.best]));
		double[] diffSigma = new double[pop[stats.worst].getSigma().length];		
		return stats;
	}

	// We draw tsize number of competitores and return the best one (maxScore)
	private int tournament(double[] fitness, int tsize) {
		int best = 0;
		double fbest = -1.0e34;

		for (int i = 0; i < tsize; i++) {
			int competitor = rd.nextInt(POPSIZE);
			if (fitness[competitor] > fbest) {
				fbest = fitness[competitor];
				best = competitor;
			}
		}
		return best;
	}

	private int mostSimilarIndiv(double[] sigma) {
		int mostSimilar = -1;
		double closestDist = 100000;
		for (int i = 0; i < POPSIZE; i++) {
			double dist = distance(population[i].getSigma(), sigma);
			if (dist < closestDist) {
				closestDist = dist;
				mostSimilar = i;
			}
		}
		return mostSimilar;
	}

	private double distance(double[] sig1, double[] sig2) {
		double distance = 0;
		for (int i = 0; i < sig1.length; i++) {
			double low = Math.min(sig1[i], sig2[i]);
			// Dividing with low will make small sigmas have more effect in
			// distance
			distance += (sig1[i] - sig2[i]) / low;
		}
		return distance;
	}

	private double[] crossover(double[] parent1, double[] parent2) {
		double[] offspring = new double[parent1.length];
		int from = rd.nextInt(parent1.length - 1);
		// ensure length>=1 so we do some crossoverlength
		int crossoverLength = Math.max(rd.nextInt(parent1.length - from), 1);
		//System.out.println("Crossover" + from + "," + crossoverLength);
		if (rd.nextDouble() > 0.5) {
			System.arraycopy(parent1, 0, offspring, 0, offspring.length);
			System.arraycopy(parent2, from, offspring, from, crossoverLength);
		} else {
			System.arraycopy(parent2, 0, offspring, 0, offspring.length);
			System.arraycopy(parent1, from, offspring, from, crossoverLength);
		}
		return offspring;
	}

	private boolean equals(double[] indiv1, double[] indiv2) {
		for (int i = 0; i < indiv1.length; i++) {
			if (indiv1[i] != indiv2[i]) {
				return false;
			}
		}
		return true;
	}

	private double[] mutate(double[] parent) {
		double[] mutation = new double[parent.length];
		int mutDest = rd.nextInt(parent.length);
		System.arraycopy(parent, 0, mutation, 0, parent.length);
		double mutValue = (rd.nextGaussian() * (MAX_SIGMA - MIN_SIGMA) * 0.4) * temperature;
		mutation[mutDest] += mutValue;
		mutation[mutDest] = Math.min(MAX_SIGMA, mutation[mutDest]);
		mutation[mutDest] = Math.max(MIN_SIGMA, mutation[mutDest]);
		return mutation;
	}

	public void printParams() {
		System.out.print("\nPOPSIZE=" + POPSIZE + // "\nDEPTH="+DEPTH+
				"\nCROSSOVER_PROB=" + CROSSOVER_PROB + "\nPMUT_PER_NODE=" + PMUT_PER_NODE + "\nGENERATIONS="
				+ GENERATIONS + "\nTOURNAMENT_SIZE=" + TOURNAMENT_SIZE + "\n----------------------------------\n");
	}

	public BasicPNN getBestIndividual() {
		int bestIndividual = 0;
		double bestScore = -1000;
		for (int i = 0; i < population.length; i++) {
			if (fitness[i] > bestScore) {
				bestScore = fitness[i];
				bestIndividual = i;
			}
		}
		return population[bestIndividual];
	}

	public void evolve(double solved) {
		evolve(GENERATIONS, solved);
	}

	public void evolve(int generationsMax, double solved) {
		ExecutorService pool = Executors.newFixedThreadPool(4);
		List<Future<Pair<BasicPNN, Double>>> futures = new ArrayList<>();
		initChart();
		//double evolveStartTime = System.currentTimeMillis();
		// double fitnessTotalTime = 0;
		int indivs;// , parent1, parent2,
		int parentIndex;

		//int fitnessSkipped = 0;
		// print_parms();
		double evalBestPop = -1000;
		System.out.println("Evolving");
		if (generation == 0)
			stats(fitness, population, generation);
		int mutations = 0;
		int crossovers = 0;
		// We use 30% of the population as the elite population
		double elitePerc = 0.3;
		double populationScore = this.populationScore(this.getElitePop(elitePerc));
		for (int gen = 0; gen < generationsMax; gen++) {
			if (evalBestPop > solved) {
				System.out.print("PROBLEM SOLVED\n");
				System.exit(0);
			}
			for (indivs = 0; indivs < POPSIZE; indivs++) {
				boolean crossover = rd.nextDouble() < CROSSOVER_PROB;
				boolean mutation = rd.nextDouble() < MUTATION_PROB;
				if (!(crossover || mutation)) {
					continue;
				}
				parentIndex = -1;
				parentIndex = tournament(fitness, TOURNAMENT_SIZE);
				double[] parentSigma = population[parentIndex].getSigma();
				double[] childSigma = new double[parentSigma.length];

				if (crossover) {
					int parent2Index = parentIndex;

					while (parent2Index == parentIndex)
						parent2Index = tournament(fitness, TOURNAMENT_SIZE);

					childSigma = crossover(parentSigma, population[parent2Index].getSigma());
					boolean eq1 = equals(childSigma, parentSigma);
					boolean eq2 = equals(childSigma, population[parent2Index].getSigma());
					if (eq1 || eq2) {
						// System.out.println("Bad crossover"+eq1+" , "+eq2);
						continue;
					}
					crossovers++;
				} else if (mutation) {
					childSigma = mutate(parentSigma);
					mutations++;
				}

				BasicPNN childPNN = clonePNN(population[parentIndex]);
				try {
					childPNN.setSigmas(childSigma);
					GeneticWorker worker = new GeneticWorker(childPNN, childPNN.getSamples(), pnnCfMatrixTrain);
					futures.add(pool.submit(worker));
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			for (Future<Pair<BasicPNN, Double>> future : futures) {
				if (stopEvolution) {
					System.out.println("Stopping Evolution Skip Futures");
					break;
				}
				try {
					BasicPNN child = future.get().getKey();
					double childFitness = future.get().getValue().doubleValue();
					int worstPop = mostSimilarIndiv(child.getSigma());
					if (fitness[worstPop] < childFitness) {
						BasicPNN revertPop = population[worstPop];
						double revertFitness = fitness[worstPop];
						population[worstPop] = child;
						fitness[worstPop] = childFitness;
						// If the population score gets worse we reject the
						// child
						// population score can get worse if no diversity in
						// population
						double candPopScore = populationScore(getElitePop(elitePerc));
						if (candPopScore > populationScore)
							persistPNN.save(getOutputStream(worstPop), population[worstPop]);
						else {
							// reject the child
							population[worstPop] = revertPop;
							fitness[worstPop] = revertFitness;
							//System.out.println("Pop Score decreased, child rejected");
						}
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			temperature *= 0.98;
			generation++;

			//System.out.println("Crossovers" + crossovers + "Mutations" + mutations);
			StatsRecord trainStats = stats(fitness, population, generation);
			evalBestPop = trainStats.bestValue;
			lineChart.addData(TRAIN_AVERAGE, generation, trainStats.averageValue);
			lineChart.addData(TRAIN_BEST, generation, trainStats.bestValue);
			lineChart.addData(TRAIN_WORST, generation, trainStats.worstValue);
			double bestValueValid = fitnessFunction(population[trainStats.best], validDataSet, pnnCfMatrixValid);
			lineChart.addData(VALID_BEST, generation, bestValueValid);
			double worstValueValid = fitnessFunction(population[trainStats.worst], validDataSet, pnnCfMatrixValid);
			lineChart.addData(VALID_WORST, generation, worstValueValid);
			try {
				saveFitness(path, description);
			} catch (Exception e) {
				e.printStackTrace();
			}
			if (stopEvolution) {
				System.out.println("Leaving Evolution");
				break;
			}
		}
		pool.shutdown();
		// Evolution done
		try {
			// saveFitness(path,description);
			System.out.println("saving pop");
			savePopulation(path, description);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//double evolveTotalTime = System.currentTimeMillis() - evolveStartTime;
		// System.out.println("Skipped fitness_function" + fitnessSkipped);
		// System.out.println("evolveTime" + evolveTotalTime);
		// System.out.print("PROBLEM *NOT* SOLVED\n");
		log.close();
	}

	private BasicPNN clonePNN(BasicPNN pnn) {
		BasicPNN clone = new BasicPNN(pnn.getKernel(), pnn.getOutputMode(), pnn.getInputCount(), pnn.getOutputCount(),
				pnn.isSeparateClass(), pnn.usePriors());
		clone.setSamples(pnn.getSamples());
		clone.setClassNames(pnn.getClassNames());
		return clone;
	}

	public void saveFitness(String dir, String descr) throws IOException {
		String filename = getFitnessFileName(dir, descr);
		DataOutputStream out = new DataOutputStream(new FileOutputStream(filename));
		out.writeInt(POPSIZE);
		try {
			for (int indiv = 0; indiv < POPSIZE; indiv++) {
				out.writeDouble(fitness[indiv]);
			}
		} catch (IOException e) {
			e.printStackTrace();
			out.close();
		}
	}

	public List<Integer> loadFitness(String dir, String descr) throws IOException {
		String filename = getFitnessFileName(dir, descr);
		DataInputStream in = new DataInputStream(new FileInputStream(filename));
		int popSize = in.readInt();
		this.fitness = new double[POPSIZE];
		List<Integer> okFitness = new ArrayList<>();
		try {
			for (int indiv = 0; indiv < popSize; indiv++) {
				fitness[indiv] = in.readDouble();
				log.println(indiv + " Load fitness=" + fitness[indiv]);
				okFitness.add(new Integer(indiv));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		in.close();
		return okFitness;
	}

	private String getFitnessFileName(String dir, String descr) {
		return dir + "/" + descr + ".fit";
	}

	public void savePopulation(String dir, String descr) throws IOException {
		for (int indiv = 0; indiv < POPSIZE; indiv++)
			persistPNN.save(getOutputStream(indiv), population[indiv]);
	}

	public void loadPopulation(String dir, String descr, BasicPNN pnn, MLDataSet dataset) throws Exception {
		List<Integer> okFitness = loadFitness(dir, descr);
		if (population == null)
			population = new BasicPNN[POPSIZE];
		for (int indiv = 0; indiv < POPSIZE; indiv++) {
			try {
				population[indiv] = (BasicPNN) persistPNN.read(getInputStream(indiv));
				population[indiv].setSamples(dataset);
				population[indiv].setClassNames(pnn.getClassNames());
			} catch (IOException e) {
				e.printStackTrace();
				System.out.println("creating random indiv" + indiv);
				population[indiv] = createRandomIndiv(pnn);				
				fitness[indiv] = fitnessFunction(population[indiv], population[indiv].getSamples(), pnnCfMatrixTrain);
				okFitness.add(new Integer(indiv));
				try {
					persistPNN.save(getOutputStream(indiv), population[indiv]);
				} catch (IOException saveIOE) {
					System.out.println("Couldnt save new random indiv" + indiv);
					saveIOE.printStackTrace();
				}
			}
			
			if (!okFitness.contains(new Integer(indiv))) {
				fitness[indiv] = fitnessFunction(population[indiv], population[indiv].getSamples(), pnnCfMatrixTrain);
				System.out.println(indiv + "new eval fitness=" + fitness[indiv]);
			}
		}
	}

	private class GeneticWorker implements Callable<Pair<BasicPNN, Double>> {
		private BasicPNN pnn;
		private MLDataSet dataset;
		Map<BasicPNN, ConfusionMatrix> pnnCFMap;

		public GeneticWorker(BasicPNN pnn, MLDataSet dataset, Map<BasicPNN, ConfusionMatrix> map) {
			this.pnn = pnn;
			this.dataset = dataset;
			this.pnnCFMap = map;
		}

		public Pair<BasicPNN, Double> call() {
			double fitness = fitnessFunction(pnn, dataset, pnnCFMap);
			return new Pair<>(pnn, new Double(fitness));
		}
	}

	private class StatsRecord {
		private int best, worst;
		private double averageValue, bestValue, worstValue;

		public StatsRecord() {
			best = 0;
			worst = 0;
			bestValue = -10000;
			worstValue = 10000;
			averageValue = 0;
		}
	}
}
