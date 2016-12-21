package util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class MyLogger {
  private File output=null;
  private PrintWriter printWriter;
  
  public MyLogger(String path) throws IOException
  {	  	  
	output = new File(path);
    printWriter = new PrintWriter(new FileWriter(output));
  }
  
  public void println(String txt)
  {
	  printWriter.println(txt);
	  printWriter.flush();
  }
  
  public void append(String txt)
  {
	  
  }
  
  public void close()
  {
	  printWriter.close();
  }
}
