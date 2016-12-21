/*
 * Encog(tm) Core v3.2 - Java Version
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-core
 
 * Copyright 2008-2013 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package neural.networks.training.pnn;

import org.encog.Encog;
import org.encog.util.EngineArray;
import org.encog.util.logging.EncogLogging;

import chart.jfree.LineChart;

import javax.swing.JFrame;


/**
 * This class determines optimal values for multiple sigmas in a PNN kernel.
 * This is done using a CJ (conjugate gradient) method.
 * 
 * Some of the algorithms in this class are based on C++ code from:
 * 
 * Advanced Algorithms for Neural Networks: A C++ Sourcebook by Timothy Masters
 * John Wiley & Sons Inc (Computers); April 3, 1995 ISBN: 0471105880
 */
public class DeriveMinimum implements CalcError {

	private TrainBasicPNN network;
	private double[] x;
	private double[] direction;
	private double minSigma;
	private LineChart lineChart;

	/**
	 * Derive the minimum, using a conjugate gradient method.
	 * 
	 * @param maxIterations
	 *            The max iterations.
	 * @param maxError
	 *            Stop at this error rate.
	 * @param eps
	 *            The machine's precision.
	 * @param tol
	 *            The convergence tolerance.
	 * @param network
	 *            The network to get the error from.
	 * @param n
	 *            The number of variables.
	 * @param x
	 *            The independent variable.
	 * @param ystart
	 *            The start for y.
	 * @param base
	 *            Work vector, must have n elements.
	 * @param direc
	 *            Work vector, must have n elements.
	 * @param g
	 *            Work vector, must have n elements.
	 * @param h
	 *            Work vector, must have n elements.
	 * @param deriv2
	 *            Work vector, must have n elements.
	 * @return The best error.
	 */
	public double calculate(final int maxIterations, final double minSigma, final double targetError, final double eps,
			final double tol, final TrainBasicPNN network, final int n, final double[] x, final double ystart){		
		this.network = network;
		this.x = x;
		this.direction = new double[x.length];
		double[] deriv2= new double[x.length];				
		this.minSigma = minSigma;
		double prevBest, toler, gam, improvement;
		final GlobalMinimumSearch globalMinimum = new GlobalMinimumSearch();
		prevBest = 1.e30;
		double fbest = network.calcErrorWithMultipleSigma(x, direction, deriv2, true);	
		for (int i = 0; i < n; i++) {
			direction[i] = -direction[i];
		}
		double[] g = new double[x.length];
		EngineArray.arrayCopy(direction, g);
		double[] h = new double[x.length];
		EngineArray.arrayCopy(direction, h);

		int convergenceCounter = 0;
		int poorCJ = 0;
		initChart();
		// Main loop
		for (int iteration = 0; iteration < maxIterations; iteration++) {			
			if (fbest < targetError) {
				System.out.println("Exit fbest < maxError" + fbest + " , " + targetError);
				break;
			}

			EncogLogging.log(EncogLogging.LEVEL_INFO, "Beginning internal Iteration #" + iteration + ", currentError="
					+ fbest + ",target=" + targetError);

			// Check for convergence
			if (prevBest <= 1.0) {
				toler = tol;
			} else {
				toler = tol * prevBest;
			}

			// Stop if there is little improvement
			if ((prevBest - fbest) <= toler) {
				if (++convergenceCounter >= 3) {
					System.out.println("exit at convergenceCounter");
					break;
				}
			} else {
				convergenceCounter = 0;
			}

			/*
			 * Here we do a few quick things for housekeeping. We save the base
			 * for the linear search in 'base', which lets us parameterize from
			 * t=0. We find the greatest second derivative. This makes an
			 * excellent scaling factor for the search direction so that the
			 * initial global search for a trio containing the minimum is fast.
			 * Because this is so stable, we use it to bound the generally
			 * better but unstable Newton scale. We also compute the length of
			 * the search vector and its dot product with the gradient vector,
			 * as well as the directional second derivative. That lets us use a
			 * sort of Newton's method to help us scale the initial global
			 * search to be as fast as possible. In the ideal case, the 't'
			 * parameter will be exactly equal to 'scale', the center point of
			 * the call to glob_min.
			 */

			double dot1 = 0; // For finding directional derivs
			double dot2 = 0; // For scaling glob_min
			double dlen = 0;

			dot1 = dot2 = dlen = 0.0;
			double high = 1.e-4;
			// i loops over sigmas
			for (int i = 0; i < n; i++) {
				//base[i] = x[i]; // For scaling glob_min
				if (deriv2[i] > high) {// Keep track of second derivatives
					high = deriv2[i];// For linear search via glob_min
				}
				dot1 += direction[i] * g[i]; // Directional first derivative
				dot2 += direction[i] * direction[i] * deriv2[i]; // and second
				dlen += direction[i] * direction[i]; // Length of search vector
			}

			dlen = Math.sqrt(dlen); // Actual length

			/*
			 * The search direction is in 'direc' and the maximum second
			 * derivative is in 'high'. That stable value makes a good
			 * approximate scaling factor. The ideal Newton scaling factor is
			 * numerically unstable. So compute the Newton ideal, then bound it
			 * to be near the less ideal but far more stable maximum second
			 * derivative. Pass the first function value, corresponding to t=0,
			 * to the routine in *y2 and flag this by using a negative npts.
			 */

			double scale;

			if (Math.abs(dot2) < Encog.DEFAULT_DOUBLE_EQUAL) {
				scale = 0;
			} else {
				scale = dot1 / dot2; // Newton's ideal but unstable scale
				//sometimes deriv2 is extreme high resulting in almost zero scale
				scale=Math.max(0.0001,scale);
			}
			high = 1.5 / high; // Less ideal but more stable heuristic
			if (high < 1.e-4) { // Subjectively keep it realistic
				high = 1.e-4;
			}

			if (scale < 0.0) { // This is truly pathological
				scale = high; // So stick with old reliable
			} else if (scale < 0.1 * high) { // Bound the Newton scale
				scale = 0.1 * high;
			} else if (scale > 10.0 * high) {
				scale = 10.0 * high;
			}
			prevBest = fbest;

			globalMinimum.setY2(fbest);
			double globMinHigh = scale * 2;// We will search for
											// improvements in the range
											// 0..gmHigh
			int numberOfPoints = 20;
			boolean firstPointKnown = false;
			globalMinimum.findBestRange(0, globMinHigh, numberOfPoints, false, targetError, this, firstPointKnown);
		
			if (globalMinimum.getY2() < fbest) {// if global caused
				// improvement
				implementImprovement(globalMinimum.getX2());
				fbest = globalMinimum.getY2();
			}

			double fBestBeforeBrentMin = fbest;
	
			if (convergenceCounter > 0) {
				fbest = globalMinimum.brentmin(20, targetError, eps, 1.e-7, this, globalMinimum.getY2());
			} else {
				fbest = globalMinimum.brentmin(10, targetError, 1.e-6, 1.e-5, this, globalMinimum.getY2());
			}
			lineChart.addData("Train error",iteration, fbest);
	
			if (fbest < fBestBeforeBrentMin) {
				implementImprovement(globalMinimum.getX2());			
			}
			
			improvement = (prevBest - fbest) / prevBest;
			
			fbest = network.calcErrorWithMultipleSigma(x, direction, deriv2, true);
			if (fbest < targetError) {
				break;
			}
			for (int i = 0; i < n; i++) {
				direction[i] = -direction[i]; // negative gradient
			}

			gam = gamma(n, g, direction);

			if (gam < 0.0) {
				gam = 0.0;
			}

			if (gam > 10.0) {
				gam = 10.0;
			}

			if (improvement < 0.001) {
				++poorCJ;
			} else {
				poorCJ = 0;
			}

			if (poorCJ >= 2) {
				if (gam > 1.0) {
					gam = 1.0;
				}
			}

			if (poorCJ >= 6) {
				poorCJ = 0;
				gam = 0.0;
			}

			findNewDir(n, gam, g, h, direction);
			if (network.validation!=null)
			{
				double validationError=network.calculateErrorWithWorkers(network.validation, false, x, null, null);
				lineChart.addData("Validation Error", iteration, validationError);
			}
			
		}
		return fbest;
	}

	private void implementImprovement(double searchpoint) {
		double[] temp = move(searchpoint, x, direction);
		EngineArray.arrayCopy(temp, x);
	}

	public double calcError(double t) {		
		double[] testSigma = move(t, x, direction);
		// When we get sigma below our minimum allowed sigma return a big error
		// so it can't be used as an solution
		for (int i = 0; i < testSigma.length; i++)
			if (testSigma[i] < minSigma)
				return 1000000;
		double error = network.calcErrorWithMultipleSigma(testSigma, null, null, false);
		return error;
	}

	protected double[] move(double t, double[] sigma, double[] direction) {
		double[] dest = new double[sigma.length];
		for (int i = 0; i < dest.length; i++) {
			dest[i] = sigma[i] + (t * direction[i]);
			dest[i] = Math.max(dest[i], this.minSigma);
		}
		return dest;
	}

	/**
	 * Find gamma.
	 * 
	 * @param n
	 *            The number of variables.
	 * @param gam
	 *            The gamma value.
	 * @param g
	 *            The "g" value, used for CJ algorithm.
	 * @param h
	 *            The "h" value, used for CJ algorithm.
	 * @param grad
	 *            The gradients.
	 */
	private void findNewDir(final int n, final double gam, final double[] g, final double[] h, final double[] grad) {
		int i;
		System.arraycopy(grad, 0, g, 0, n);
		for (i = 0; i < n; i++) {
			grad[i] = h[i] = g[i] + gam * h[i];
		}
	}

	/**
	 * Find correction for next iteration.
	 * 
	 * @param n
	 *            The number of variables.
	 * @param g
	 *            The "g" value, used for CJ algorithm.
	 * @param grad
	 *            The gradients.
	 * @return The correction for the next iteration.
	 */
	private double gamma(final int n, final double[] g, final double[] grad) {
		int i;
		double denom, numer;

		numer = denom = 0.0;

		for (i = 0; i < n; i++) {
			denom += g[i] * g[i];
			numer += (grad[i] - g[i]) * grad[i]; // Grad is neg gradient
		}

		if (denom == 0.0) {
			return 0.0;
		} else {
			return numer / denom;
		}
	}
	
	private JFrame initChart() {
		this.lineChart = new LineChart("DeriveMinimum Training", "Iteration", "Error");
		JFrame frame = new JFrame();
		frame.setSize(1400, 1200);
		frame.getContentPane().add(lineChart);
		frame.pack();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		return frame;
	}
}
