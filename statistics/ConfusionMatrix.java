package statistics;

import java.awt.Color;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import util.MathUtil;

public class ConfusionMatrix extends JPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 24563412364561L;
	private String description;
	private String[] names;
	private int[][] confusion;
	private double[][] scores;
	private JLabel[][] gridLabels;
	//private JLabel[] typeIIError;
	//private JLabel[] typeIError;
	private JLabel trueRatio;
	private JLabel descriptionLabel;
	public ConfusionMatrix(String[] names) {
		this(names, "Confusion Matrix");
	}

	public ConfusionMatrix(String[] names, String description) {
		this.names = names;
		int nrOfClasses = names.length;
		this.confusion = new int[nrOfClasses][nrOfClasses];
		this.scores = new double[nrOfClasses][nrOfClasses];
		initScores();
		resetConfusion();
		this.description = description;
		initPanel();
	}

	public void updateGrid() {
		for (int row = 0; row < names.length; row++) {
		
			for (int col = 0; col < names.length; col++) {
				this.gridLabels[row][col].setText(Integer.toString(getVal(row, col)));
			}
			//double error = MathUtil.nDecimals(this.getFNR(row), 3);
			//this.typeIIError[row].setText(Double.toString(error));
		}
		
		/*
		for (int col = 0; col < names.length; col++) {
			double error = MathUtil.nDecimals(this.getFPR(col), 3);
			this.typeIError[col].setText(Double.toString(error));
		}*/		
		this.trueRatio.setText("Correct Classification Ratio : "+MathUtil.nDecimals(this.getTrueRatio(),3));
		this.descriptionLabel.setText(this.description);
	}

	private void initPanel() {
		this.setLayout(new GridBagLayout());
		GridBagConstraints c = new GridBagConstraints();
		c.gridx = 0;
		c.gridy = 0;
		c.gridwidth=2;
		this.descriptionLabel = new JLabel(this.description);
		//this.descriptionLabel.setHorizontalAlignment(SwingConstants.CENTER);
		this.add(descriptionLabel, c);
		c.gridwidth=1;
		c.gridx = 1;
		c.gridy = 1;
		c.fill = GridBagConstraints.BOTH;
		JLabel predictLabel= new JLabel("Predicted Class");
		predictLabel.setHorizontalAlignment(SwingConstants.CENTER);
		this.add(predictLabel,c);
		JPanel actualPanel = new JPanel();
		actualPanel.setLayout(new GridLayout(5, 1));
		actualPanel.add(new JLabel(""));
		actualPanel.add(new JLabel(""));
		actualPanel.add(new JLabel("Actual"));
		actualPanel.add(new JLabel("Class"));
		c.gridy++;
		c.gridx = 0;
		this.add(actualPanel, c);
		c.gridx = 1;
		this.add(createConfusionGrid(), c);
		c.gridy++;
		c.gridx = 0;
		c.gridwidth = 2;
		this.add(trueRatio=new JLabel("True ratio : " + this.getTrueRatio()), c);
		trueRatio.setHorizontalAlignment(SwingConstants.CENTER);
	}

	private JPanel createConfusionGrid() {
		JPanel grid = new JPanel();
		GridBagLayout layout = new GridBagLayout();
		grid.setLayout(layout);
		GridBagConstraints c = new GridBagConstraints();
		c.gridx = 1;
		c.gridy = 0;
		c.fill = GridBagConstraints.BOTH;
		this.gridLabels = new JLabel[names.length][names.length];
		JLabel[] labelNames = new JLabel[names.length];

		for (int i = 0; i < names.length; i++) {
			labelNames[i] = new JLabel(names[i]);
			labelNames[i].setHorizontalAlignment(SwingConstants.CENTER);
			c.gridx = i + 1;
			grid.add(labelNames[i], c);
		}

		c.gridx++;
		//JLabel label = new JLabel("FNR");
		//grid.add(label);
		//this.typeIIError = new JLabel[names.length];

		for (int row = 0; row < names.length; row++) {
			c.gridy++;
			c.gridx = 0;
			grid.add(new JLabel(labelNames[row].getText()), c);

			for (int col = 0; col < names.length; col++) {
				c.gridx++;
				this.gridLabels[row][col] = new JLabel(Integer.toString(getVal(row, col)));
				this.gridLabels[row][col].setBorder(BorderFactory.createLineBorder(Color.BLACK));
				this.gridLabels[row][col].setHorizontalAlignment(SwingConstants.CENTER);
				grid.add(this.gridLabels[row][col], c);
			}
			//c.gridx++;
			//this.typeIIError[row] = new JLabel("fnr");
			//grid.add(this.typeIIError[row], c);
		}
		/*
		c.gridy++;
		c.gridx=0;
		this.typeIError=new JLabel[names.length];
		grid.add(new JLabel("FPR"),c);
		for (int col = 0; col < names.length; col++) {	
			c.gridx++;
			this.typeIError[col]=new JLabel("fpr");
			grid.add(this.typeIError[col],c);
			
		}*/
		grid.setBorder(new javax.swing.border.BevelBorder(1));
		return grid;
	}

	public void initScores() {
		int nrOfClasses = names.length;
		for (int row = 0; row < nrOfClasses; row++) {
			for (int col = 0; col < nrOfClasses; col++) {
				scores[row][col] = row == col ? 1 : 0;
			}
		}
	}

	public void setScoreMatrix(double val, int row, int col) {
		scores[row][col] = val;
	}

	public void setScoreMatrix(double[][] newScoreMatrix) {
		int nrOfClasses = names.length;
		for (int row = 0; row < nrOfClasses; row++)
			for (int col = 0; col < nrOfClasses; col++)
				scores[row][col] = newScoreMatrix[row][col];
	}

	public double[][] getScoreMatrix() {
		return scores;
	}

	public double getScore() {
		double score = 0;
		int nrOfClasses = names.length;
		for (int row = 0; row < nrOfClasses; row++) {
			for (int col = 0; col < nrOfClasses; col++) {
				score += confusion[row][col] * scores[row][col];
			}
		}
		return score;
	}

	public double getScore(ConfusionMatrix scoreMatrix) {
		double score = 0;
		int nrOfClasses = names.length;
		for (int row = 0; row < nrOfClasses; row++) {
			for (int col = 0; col < nrOfClasses; col++) {
				double scoreRC = confusion[row][col] * scoreMatrix.scores[row][col];
				score += scoreRC;
			}
		}
		return score;
	}

	public void update(int[] actualClasses, int[] predictedClasses) {
		resetConfusion();
		for (int i = 0; i < predictedClasses.length; i++) {
			confusion[actualClasses[i]][predictedClasses[i]]++;
		}
	}

	public void resetConfusion() {
		int nrOfClasses = names.length;
		for (int row = 0; row < nrOfClasses; row++) {
			for (int col = 0; col < nrOfClasses; col++)
				confusion[row][col] = 0;
		}
	}

	public String[] getNames() {
		return names;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public String getClassDescriptions() {
		String temp = "Cl" + names[0];
		for (int i = 1; i < names.length; i++) {
			temp += "," + names[i];
		}
		return temp;
	}

	public String getScoreDescription() {
		String scoreDescr = "Scores";
		for (int row = 0; row < scores.length; row++)
			for (int col = 0; col < scores.length; col++) {
				if (scores[row][col] != 0)
					scoreDescr += "{" + row + "," + col + "," + scores[row][col] + "}";
			}
		return scoreDescr;
	}

	public int getColumns() {
		return names.length;
	}

	public int getVal(int actual, int predicted) {
		return confusion[actual][predicted];
	}

	public static int getClass(double value, double[] classRanges) {
		int classId = -1;
		for (int i = 0; i < classRanges.length; i++) {
			if (value < classRanges[i]) {
				classId = i;
				break;
			}
		}
		if (classId == -1)
			System.out.println(value);
		return classId;
	}

	public void increase(int actual, int predicted) {
		confusion[actual][predicted]++;
	}

	public void increase(int actual, int predicted, double value) {
		confusion[actual][predicted]++;
	}

	public int[][] getConfusionconfusion() {
		return confusion;
	}

	// All correctly classified patterns
	public int getPositives() {
		int pos = 0;
		for (int i = 0; i < names.length; i++)
			pos += confusion[i][i];
		return pos;
	}

	// All patterns that are not correctly classified
	public int getNegatives() {
		int neg = 0;
		for (int row = 0; row < names.length; row++)
			for (int col = 0; col < names.length; col++) {
				if (row != col)
					neg += confusion[row][col];
			}
		return neg;
	}

	// All patterns that doesn't belong to class actual
	public int getNegatives(int actual) {
		int neg = 0;
		for (int row = 0; row < names.length; row++)
		{
			if (row==actual)
				continue;
			for (int col = 0; col < names.length; col++) {
				neg += confusion[row][col];
			}
		}
		return neg;
	}

	// All correct rejections from actual class
	public int getTrueNegatives(int actual) {
		int tn = 0;
		for (int row = 0; row < names.length; row++)
			for (int col = 0; col < names.length; col++) {
				if (row == actual || col == actual)
					continue;
				tn += confusion[row][col];
			}
		return tn;
	}

	/*
	 * Type I error from the actual class perspective
	 */
	public int getFalsePositives(int actual) {
		int fp = 0;
		for (int row = 0; row < names.length; row++) {
			if (row != actual)
				fp += confusion[row][actual];
		}
		return fp;
	}

	/*
	 * Type II error , The positive (pattern belongs to actual) is falsely
	 * classified as negative
	 */
	public int getFalseNegatives(int actual) {
		int fn = 0;
		for (int col = 0; col < names.length; col++) {
			if (col != actual)
				fn += confusion[actual][col];
		}
		return fn;
	}

	/*
	 * The number correct classification for class actual/total patterns of
	 * class actual
	 */
	public double getTPR(int actual) {
		return ((double) confusion[actual][actual]) / getRowSum(actual);
	}

	/*
	 * Type II error ratio
	 */
	public double getFNR(int actual) {
		return 1 - getTPR(actual);
	}

	/*
	 * How often do we correctly classify a pattern that is class actual/ (total
	 * pattern - total patterns of class actual)
	 */

	public double getTNR(int actual) {
		return ((double) getTrueNegatives(actual)) / getNegatives(actual);
	}

	/*
	 * Type I error ratio
	 */
	public double getFPR(int actual) {
		return 1 - getTNR(actual);
	}

	public double getTrueRatio() {
		double truePos = 0;
		int all = 0;
		for (int row = 0; row < names.length; row++)
			for (int col = 0; col < names.length; col++) {
				if (row == col)
					truePos += confusion[row][col];
				all += confusion[row][col];
			}
		return truePos / (double) all;
	}

	public int getNrOfPatterns() {
		int sum = 0;
		for (int row = 0; row < names.length; row++)
			for (int col = 0; col < names.length; col++) {
				sum += confusion[row][col];
			}
		return sum;
	}

	private int getRowSum(int row) {
		int sum = 0;
		for (int col = 0; col < names.length; col++) {
			sum += confusion[row][col];
		}
		return sum;
	}

	public void add(ConfusionMatrix cm) {
		int nrOfClasses = names.length;
		for (int row = 0; row < nrOfClasses; row++)
			for (int col = 0; col < nrOfClasses; col++) {
				confusion[row][col] += cm.getVal(row, col);
			}
	}

	public double[][] getConfusionMatrixPerc() {
		int nrOfClasses = names.length;
		double[][] perc = new double[nrOfClasses][nrOfClasses];

		for (int row = 0; row < nrOfClasses; row++) {
			int rowSum = getRowSum(row);
			for (int col = 0; col < nrOfClasses; col++) {
				perc[row][col] = confusion[row][col] / (double) rowSum;
			}
		}
		return perc;
	}

	@Override
	public ConfusionMatrix clone() {
		ConfusionMatrix clone = new ConfusionMatrix(names);
		clone.add(this);
		return clone;
	}

	public String toString() {
		String result = "";
		result += "Number of patterns = " + getNrOfPatterns() + "\n";
		result += "Correct Classification Ratio=" + MathUtil.nDecimals(getTrueRatio(), 3) + "\n";
		for (int col = 0; col < names.length; col++)
			result += names[col] + ",";
		result += "\n";
		double[][] confPerc = getConfusionMatrixPerc();
		int[] colPatterns = new int[names.length];
		int colWidth = 8;
		for (int row = 0; row < names.length; row++) {
			int rowPatterns = 0;
			String line = "";
			for (int col = 0; col < names.length; col++) {
				String colStr = "";
				rowPatterns += confusion[row][col];
				colPatterns[col] += confusion[row][col];
				boolean showPatterns = true;
				if (showPatterns)
					colStr += confusion[row][col];
				else
					colStr += MathUtil.nDecimals(confPerc[row][col], 3);
				while (colStr.length() < colWidth)
					colStr += " ";
				String data = "(" + colStr + ")";
				if (col == 0)
					line += data;
				else
					line += " , " + data;
			}
			double rowPerc = MathUtil.nDecimals(rowPatterns / (double) getNrOfPatterns(), 3);
			result += line + "Actual Perc" + rowPerc + "\n";
		}
		return result;
	}
}
