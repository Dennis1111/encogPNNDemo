package examples.pnn;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;

public class AbaloneReader {
	
	public static BasicMLDataSet readAbalone() {
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
			while (fileScanner.hasNextLine()) {
				List<String> tokens = ScannerUtil.processLine(fileScanner.nextLine(),",");			
				if (tokens.size()==0)
					break;
				double[] inputs = new double[8];
				// sex
				inputs[0] = convertSex(tokens.get(0));
				// length
				inputs[1] = doubleValue(tokens.get(1));
				// diameter
				inputs[2] = doubleValue(tokens.get(2));
				// height
				inputs[3] = doubleValue(tokens.get(3));
				//whole height
				inputs[4] = doubleValue(tokens.get(4));
				//shucked weight
				inputs[5] = doubleValue(tokens.get(5));
				//viscera weight
				inputs[6] = doubleValue(tokens.get(6));
				//shell weight
				inputs[7] = doubleValue(tokens.get(7));
				
				double[] ideals = new double[1];
				ideals[0]=convertRings(tokens.get(8));				
				//System.out.println("input" + Arrays.toString(inputs) + " ideals" + Arrays.toString(ideals));
				samples.add(new BasicMLData(inputs), new BasicMLData(ideals));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Samples"+samples.size());
		return samples;
	}
	
	//Class 0 if rings 1-8, Class 1 ring 9-10, 11.. Class2
	private static double convertRings(String rings){
	  	int ringsAsInt=(int) doubleValue(rings);
	  	if (ringsAsInt<=8)
	  		return 0;
	  	else if (ringsAsInt<=10)
	  		return 1;
	  	else
	  		return 2;	  	
	}
	
	private static double convertSex(String sex){
		if (sex.equals("M"))
			return 0;
		else if (sex.equals("F")) 
			return 1;
		else return 2;
	}
	
	private static double doubleValue(String token) {
		return new Double(token).doubleValue();
	}
	
	public static void main(String[] args){
		readAbalone();
	}
}
