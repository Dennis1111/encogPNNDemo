/* ---------------------
 * SampleXYDataset2.java
 * ---------------------
 * (C) Copyright 2000-2014, by Object Refinery Limited.
 * 
 * http://www.object-refinery.com
 * 
 */

package chart.jfree;

import org.jfree.data.DomainInfo;
import org.jfree.data.Range;
import org.jfree.data.RangeInfo;
import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYSeries;

/**
 * 
 */
public class MyXYDataset extends AbstractXYDataset 
  implements XYDataset, DomainInfo, RangeInfo {

  /**
	 * 
	 */
	private static final long serialVersionUID = 456156346L;

/** The series count. */
  private static final int DEFAULT_SERIES_COUNT = 4;
  
  /** The range. */
  private static final double DEFAULT_RANGE = 200;
  
  private XYSeriesCollection xySeriesCollection;
  
  /** The minimum domain value. */
  private Number domainMin;
  
  /** The maximum domain value. */
  private Number domainMax;
  
  /** The minimum range value. */
  private Number rangeMin;
  
  /** The maximum range value. */
  private Number rangeMax;
  
  /** The range of the domain. */
  private Range domainRange;
  
  /** The range. */
  private Range range;

  private String name;
  
  /**
   * Creates a sample dataset.
   *
   * @param seriesCount  the number of series.
   * @param itemCount  the number of items.
   */
  public MyXYDataset() 
  {
    
    xySeriesCollection = new XYSeriesCollection();
    double minX = -10;
    double maxX = 10;
    double minY = -10;
    double maxY = 10;
    
    
    this.domainMin = new Double(minX);
    this.domainMax = new Double(maxX);
    this.domainRange = new Range(minX, maxX);
    
    this.rangeMin = new Double(minY);
    this.rangeMax = new Double(maxY);
    this.range = new Range(minY, maxY);	
    this.name="";
  }

  public void setName(String name)
  {
    this.name=name;
  }
  
  public String getName()
  {
    return name;
  }
  public void addSeries(XYSeries series)
  {
    xySeriesCollection.addSeries(series);
  }
  
  public void addData(double x,double y,java.lang.Comparable<String> className)
  {
    XYSeries serie = xySeriesCollection.getSeries(className);
    serie.add(x,y);
  }

  /**
   * Returns the x-value for the specified series and item.  Series are numbered 0, 1, ...
   *
   * @param series  the index (zero-based) of the series.
   * @param item  the index (zero-based) of the required item.
   *
   * @return the x-value for the specified series and item.
   */
  @Override
  public Number getX(int series, int item) {
    return this.xySeriesCollection.getX(series,item);
  }

    /**
     * Returns the y-value for the specified series and item.  Series are numbered 0, 1, ...
     *
     * @param series  the index (zero-based) of the series.
     * @param item  the index (zero-based) of the required item.
     *
     * @return  the y-value for the specified series and item.
     */
    @Override
    public Number getY(int series, int item) {
      return this.xySeriesCollection.getY(series,item);
    }

    /**
     * Returns the number of series in the dataset.
     *
     * @return the series count.
     */
    @Override
    public int getSeriesCount() {
      return this.xySeriesCollection.getSeriesCount();
    }

     /**
     * Returns the key for the series.
     *
     * @param series  the index (zero-based) of the series.
     *
     * @return The key for the series.
     */
    @Override
    public Comparable<String> getSeriesKey(int series) {
      return (String)this.xySeriesCollection.getSeriesKey(series);
    }

    /**
     * Returns the number of items in the specified series.
     *
     * @param series  the index (zero-based) of the series.
     *
     * @return the number of items in the specified series.
     */
    @Override
    public int getItemCount(int series) {
      return this.xySeriesCollection.getItemCount(series);
    }

    /**
     * Returns the minimum domain value.
     *
     * @return The minimum domain value.
     */
    public double getDomainLowerBound() {
        return this.domainMin.doubleValue();
    }

    /**
     * Returns the lower bound for the domain.
     * 
     * @param includeInterval  include the x-interval?
     * 
     * @return The lower bound.
     */
    @Override
    public double getDomainLowerBound(boolean includeInterval) {
        return this.domainMin.doubleValue();
    }
    
    /**
     * Returns the maximum domain value.
     *
     * @return The maximum domain value.
     */
    public double getDomainUpperBound() {
        return this.domainMax.doubleValue();
    }

    /**
     * Returns the upper bound for the domain.
     * 
     * @param includeInterval  include the x-interval?
     * 
     * @return The upper bound.
     */
    @Override
    public double getDomainUpperBound(boolean includeInterval) {
        return this.domainMax.doubleValue();
    }
    
    /**
     * Returns the range of values in the domain.
     *
     * @return the range.
     */
    public Range getDomainBounds() {
        return this.domainRange;
    }

    /**
     * Returns the bounds for the domain.
     * 
     * @param includeInterval  include the x-interval?
     * 
     * @return The bounds.
     */
    @Override
    public Range getDomainBounds(boolean includeInterval) {
        return this.domainRange;
    }
    
    /**
     * Returns the range of values in the domain.
     *
     * @return the range.
     */
    public Range getDomainRange() {
        return this.domainRange;
    }

    /**
     * Returns the minimum range value.
     *
     * @return The minimum range value.
     */
    public double getRangeLowerBound() {
        return this.rangeMin.doubleValue();
    }
    
    /**
     * Returns the lower bound for the range.
     * 
     * @param includeInterval  include the y-interval?
     * 
     * @return The lower bound.
     */
    @Override
    public double getRangeLowerBound(boolean includeInterval) {
        return this.rangeMin.doubleValue();
    }

    /**
     * Returns the maximum range value.
     *
     * @return The maximum range value.
     */
    public double getRangeUpperBound() {
        return this.rangeMax.doubleValue();
    }

    /**
     * Returns the upper bound for the range.
     * 
     * @param includeInterval  include the y-interval?
     * 
     * @return The upper bound.
     */
    @Override
    public double getRangeUpperBound(boolean includeInterval) {
        return this.rangeMax.doubleValue();
    }
    
    /**
     * Returns the range of values in the range (y-values).
     *
     * @param includeInterval  include the y-interval?
     * 
     * @return The range.
     */
    @Override
    public Range getRangeBounds(boolean includeInterval) {
        return this.range;
    }
    
    /**
     * Returns the range of y-values.
     * 
     * @return The range.
     */
    public Range getValueRange() {
        return this.range;
    }
    
    /**
     * Returns the minimum domain value.
     * 
     * @return The minimum domain value.
     */
    public Number getMinimumDomainValue() {
        return this.domainMin;
    }
    
    /**
     * Returns the maximum domain value.
     * 
     * @return The maximum domain value.
     */
    public Number getMaximumDomainValue() {
        return this.domainMax;
    }
    
  /**
   * Returns the minimum range value.
   * 
   * @return The minimum range value.
   */
  public Number getMinimumRangeValue() {
        return this.domainMin;
    }
    
  /**
   * Returns the maximum range value.
   * 
   * @return The maximum range value.
   */
  public Number getMaximumRangeValue() {
    return this.domainMax;
  }  
}
