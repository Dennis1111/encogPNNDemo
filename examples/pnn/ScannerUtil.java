package examples.pnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class ScannerUtil {
	public static List<String> processLine(String aLine,String delim) {
		Scanner scanner = new Scanner(aLine);
		scanner.useDelimiter(delim);
		List<String> tokens = new ArrayList<String>();
		while (scanner.hasNext()) {
			String token = scanner.next().trim();
			System.out.println(token);
			tokens.add(token);
		}
		scanner.close();
		return tokens;
	}
}
