package examples.pnn;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;

public class IrisReader {
	
	private String[] classNames={"Iris-setosa","Iris-versicolor","Iris-virginica"};
	private String[] columnNames;
	
	public String[] getClassNames(){
		return classNames;
	}
	public String[] getColumnNames(){
		return columnNames;
	}
	
	
	public BasicMLDataSet readIris() {
		   JFileChooser chooser = new JFileChooser();
		    FileNameExtensionFilter filter = new FileNameExtensionFilter(
		        "data & csv", "data","csv");
		    chooser.setFileFilter(filter);
		    JFrame frame= new JFrame();
		    int returnVal = chooser.showOpenDialog(frame);
		    if(returnVal == JFileChooser.APPROVE_OPTION) {
		       System.out.println("You chose to open this file: " +
		            chooser.getSelectedFile().getName());
		    }
			BasicMLDataSet samples = new BasicMLDataSet();
			Scanner fileScanner;
		try {
			File file=chooser.getSelectedFile();
			fileScanner = new Scanner(new FileReader(file));
			int row=0;
			while (fileScanner.hasNextLine()) {
				List<String> tokens = ScannerUtil.processLine(fileScanner.nextLine(),",");
				if (row == 0) {
					this.columnNames=tokens.toArray(new String[0]);
					row++;
					continue;
				}
				if (tokens.size() == 0)
					break;
				double[] inputs = new double[4];
				// sepalLength
				inputs[0] = doubleValue(tokens.get(0));
				// sepalWidth
				inputs[1] = doubleValue(tokens.get(1));
				// petalLength
				inputs[2] = doubleValue(tokens.get(2));
				// petalWidth
				inputs[3] = doubleValue(tokens.get(3));
				double[] ideals = new double[1];
				ideals[0]=getClass(tokens.get(4));
				System.out.println("input" + Arrays.toString(inputs) + " ideals" + Arrays.toString(ideals));
				samples.add(new BasicMLData(inputs), new BasicMLData(ideals));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Samples"+samples.size());
		return samples;
	}
	
	private int getClass(String token)
	{
		if (token.equals(classNames[0]))
			return 0;
		else if (token.equals(classNames[1]))
			return 1;
		else return 2;
			
	}
	private static double doubleValue(String token) {
		return new Double(token).doubleValue();
	}
	
	public static void main(String[] args){
		IrisReader reader= new IrisReader();
		reader.readIris();
	}
}
