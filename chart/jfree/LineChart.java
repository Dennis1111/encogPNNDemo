package chart.jfree;

import java.awt.Color;
import java.util.HashMap;
import java.util.Map;
import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RectangleInsets;

public class LineChart extends JPanel {
	/**
	 * 
	 */
	
	private static final long serialVersionUID = 452351L;
	private Map<String, XYSeries> xySeries;
	XYSeriesCollection dataset;

	/**
	 * @param title the frame title.
	 */
	public LineChart(String title, String x, String y) {
		dataset = new XYSeriesCollection();
		JFreeChart chart = createChart(title, dataset, x, y);
		ChartPanel chartPanel = new ChartPanel(chart);
		chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
		this.add(new ChartPanel(chart));
		xySeries = new HashMap<>();
	}

	public void addSeries(String name) {
		XYSeries series = new XYSeries(name);
		dataset.addSeries(series);
		xySeries.put(name, series);
	}

	public void addData(String name, double index, double value) {
		if (xySeries.get(name) == null)
			addSeries(name);
		xySeries.get(name).add(index, value);
	}

	/**
	 * Creates a chart.
	 *
	 * @param dataset
	 *            the data for the chart.
	 *
	 * @return a chart.
	 */
	private static JFreeChart createChart(String title, XYDataset dataset, String x, String y) {
		JFreeChart chart = ChartFactory.createXYLineChart(title, // chart title
				x, // x axis label
				y, // y axis label
				dataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
		);
		chart.setBackgroundPaint(Color.white);
		XYPlot plot = (XYPlot) chart.getPlot();
		plot.setBackgroundPaint(Color.lightGray);
		plot.setAxisOffset(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
		plot.setDomainGridlinePaint(Color.white);
		plot.setRangeGridlinePaint(Color.white);
		// change the auto tick unit selection to integer units only...
		//NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		//rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		return chart;
	}
}
