package util;

import java.lang.Math;
import java.util.*;

public class MathUtil
{ 
  public static Double nDecimals(Double d,double n)
  {
    return new Double(nDecimals(d.doubleValue(),n));
  }
  
  public static double nDecimals(double val,double n)
  {
    double result;
    double scaleUp=java.lang.Math.pow(10.0,n);
    val=java.lang.Math.round(val*scaleUp);
    result=val/scaleUp;
    return result;
  }
}
