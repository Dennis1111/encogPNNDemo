# encogPNNDemo
A version of Encog PNN for bugfixes and some related demos
Have temporarily removed org.encog from encog package names to keep things separate when working
on local version.

The demos Iris and Abalone depends on
Encog 3.3, EJML0.30, JFreeChart  

EJML is used for principal components and viewing the dataset
download EJML 0.30 from http://www.ejml.org and include paths to
main/dense64/src;
main/core/src;
examples/src;
download JFreeChart from http://www.jfree.org/jfreechart/
and include their paths (jfreechart and jcommon)
also download abalone.data from https://archive.ics.uci.edu/ml/datasets/Abalone
iris.csv can be found in the encog core distribution

The democlasses to run is 
examples/pnn/Iris.java and
examples/pnn/Abalone.java
The demos is hardcoded to save Population created by GeneticAlgoritmBasicPNN in "C:/MLDataSet/Iris/GA" and "C:/MLDataSet/Abalone/GA"

Should perhaps have normalized inputs but unless theres is huge difference in variance for inputs I think it's ok.

Major Changes to the encog classes
BasicPNN
1. If using separate classes is gonna work the number of sigmas should be inputcount*outputcount
2. Instead of using the excludeCounter I use isSamePattern(pattern1,pattern2), the purpose was to make it easier to implement threaded use of compute(MLData)
3. Have created a method public ConfusionMatrix computeConfusionMatrix(MLDataSet dataset) , thinks it's always nice to see the confusionMatrix for classification examples and I then use this method for training the pnn with Genetic Algoritm
4. Though i haven't test using priors i don't see how it is possible so i added 	public void setPriors(double[] priors)t
5. Experimented some with subsets and haven't cleaned it totatally up.

TrainBasicPNN
1. I don't compute derivate directly here, instead I have created a workerclass for threads
2. In iteration it chooses either to search with single or multiple sigma (this way one can avoid single sigma overwriting current sigma when one already have a descent solution)
3. Removed lots of parameters to DeriveMinimum
4. Create	public double calcError(double sigma) so when TrainBasicPNN is passed to GlobalMinimum single sigma solution will calculated

DeriveMinimum
Have mentioned the most important changes in debug forum..

TrainBasicPNNWorker
Have put the error and derivative code here but I have not been interesting in GRNN so I might have removed some code don't remember exactly.
Splitted the derivate calculation in into two functions dependent on separate classes or not.

GlobalMinimum
Not hardcoded to have sigmas as parameter
