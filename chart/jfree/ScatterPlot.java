package chart.jfree;

import java.awt.Color;

import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class ScatterPlot extends ApplicationFrame
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 23423434L;

	public ScatterPlot(String title, XYDataset xyDataset,String x,String y) {
		super(title);
		JPanel chartPanel = createPanel(title,xyDataset,x,y);
		chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);
        pack();
        RefineryUtilities.centerFrameOnScreen(this);
        setVisible(true);
	}
	
	/**
     * @return A panel.
     */
    public static JPanel createPanel(String title,XYDataset dataset,String x,String y) {
        
        JFreeChart chart = ChartFactory.createScatterPlot(title, x, y, dataset, PlotOrientation.VERTICAL, true, true, false);
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setRangeTickBandPaint(new Color(200, 200, 100, 100));
        XYDotRenderer renderer = new XYDotRenderer();
        renderer.setDotWidth(4);
        renderer.setDotHeight(4);
        plot.setRenderer(renderer);
        plot.setDomainCrosshairVisible(true);
        plot.setRangeCrosshairVisible(true);

        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setAutoRangeIncludesZero(false);
        ChartPanel panel = new ChartPanel(chart);
        panel.setMouseWheelEnabled(true);
        return panel;
    }
}
