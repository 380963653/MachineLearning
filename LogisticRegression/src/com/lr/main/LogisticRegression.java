package com.lr.main;

public class LogisticRegression {
	
	public LogisticRegression() {
		
	}
	
	/**
	 * sigmoid函数
	 * @param z
	 * @return
	 */
	public double sigmoid(double z) {
		return 1 / (1 + Math.pow(Math.E, z));
	}
	
	/**
	 * 假设函数
	 * @param theta
	 * @param x
	 * @param n
	 * @return
	 */
	public double hTheta(double[] theta, double[] x, int n) {
		double sum = 0.0;
		for(int i=0; i<n+1; i++) {
			sum += theta[i] * x[i];
		}
		
		return sigmoid(-sum);
	}
	
	/**
	 * cost function
	 * @param theta
	 * @param x
	 * @param y
	 * @param m
	 * @param n
	 * @return
	 */
	public double jTheta(double[] theta, double[][] x, double[] y, int m, int n) {
		double sum = 0.0;
		for(int i=0; i<m; i++) {
			sum += y[i] * Math.log(hTheta(theta, x[i], n)) + (1-y[i]) * Math.log(1-hTheta(theta, x[i], n));
		}
		
		return -sum / m;
	}
	
	/**
	 * 数组赋值函数
	 * @param source
	 * @param dest
	 */
	private void assignment(double[] source, double[] dest) {
		for(int i=0; i<source.length; i++) {
			dest[i] = source[i];
		}
	}
	
	/**
	 * 梯度下降函数(无正则化)
	 * @param theta
	 * @param x
	 * @param y
	 * @param m
	 * @param n
	 * @param alpha
	 * @param iterations
	 * @return
	 */
	public void gradientDescent(double[] theta, double[][] x, double[] y, int m, int n, double alpha, int iterations) {
		for(int i=0; i<iterations; i++) {
			double[] thetaTemp = new double[theta.length];
			assignment(theta, thetaTemp);
			for(int j=0; j<n+1; j++) {
				double sum = 0.0;
				for(int k=0; k<m; k++) {
					sum += (hTheta(theta, x[k], n) - y[k]) * x[k][j];
				}
				thetaTemp[j] = thetaTemp[j] - alpha * sum / m;
			}
			assignment(thetaTemp, theta);
		}
	}
	
	/**
	 * 特征缩放
	 * @param data
	 * @param m
	 * @param n
	 */
	public void featureScale(double[][] data, int m, int n) {
		double[] mean = new double[n+1];
		for(int j=0; j <n+1; j++) {
			for(int i=0; i<m; i++) {
				mean[j] += data[i][j];
			}
		}
		
		for(int j=0; j<n+1; j++) {
			mean[j] /= m;
		}
		
		double[] stddeviation = new double[n+1];
		for(int j=1; j<n+1; j++) {
			for(int i=0; i<m; i++) {
				stddeviation[j] += Math.pow(data[i][j] - mean[j], 2);
			}
		}
		
		for(int j=1; j<n+1; j++) {
			stddeviation[j] = Math.sqrt(stddeviation[j] / m);
		}
		
		for(int j=1; j<n+1; j++) {
			for(int i=0; i<m; i++) {
				data[i][j] = (data[i][j] - mean[j]) / stddeviation[j];
			}
		}
	}
	
	public void test() {
		double[][] x = {{3,3}, {4,3}, {1,1}};
		double[] y = {1, 1, -1};
		int m = 3;
		int n = 2;
		double[] initialTheta = new double[n+1];
		double learningRate = 0.001;
		int iterations = 7000;
		double[][] X = new double[m][n+1];
		for(int i=0; i<m; i++) {
			X[i][0] = 1;
			for(int j=1; j<n+1; j++) {
				X[i][j] = x[i][j-1];
			}
		}
		
		//featureScale(X, m, n);
		double[] finalTheta = new double[n+1];
		gradientDescent(finalTheta, X, y, m, n, learningRate, iterations);
		for(int i=0; i<finalTheta.length; i++) {
			System.out.println("x[" + i + "]" + "=" +finalTheta[i]);
		}
		System.out.println("Initial cost:" + jTheta(initialTheta, X, y, m, n));
		System.out.println("Final cost:" + jTheta(finalTheta, X, y, m, n));
	}
	
	public static void main(String[] args) {
		LogisticRegression lr = new LogisticRegression();
		lr.test();
	}
}
