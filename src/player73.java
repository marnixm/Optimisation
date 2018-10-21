
import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.*;
import java.lang.Math;

public class player73 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
	private int evaluations_limit_;
	SphereEvaluation sphere = new SphereEvaluation();
	boolean isMultimodal, hasStructure, isSeparable, isRegular;

	Integer INIT_POP_SIZE = 200; //100, 200
	Integer INIT_PHEN_SIZE = 10;
	Integer NUMBER_OF_PARENTS = 250; //50, 250
	Integer DECREASE_POP_SIZE_AFTER = 10;
	double INIT_MUT_RATE = 0.25; //0.25
	double INIT_MUT_STEPSIZE = 0.1; //0.1


	public player73()
	{
		rnd_ = new Random();
	}

	public static void main(String args[]){
		new player73().run();
	}

	public void setSeed(long seed)
	{
		// Set seed of algorithms random process
		rnd_.setSeed(seed);
	}

	@SuppressWarnings("unused")
	public void setEvaluation(ContestEvaluation evaluation)
	{
		System.out.println("set Evaluation");
		// Set evaluation problem used in the run
		evaluation_ = evaluation;

		// Get evaluation properties
		Properties props = evaluation.getProperties();
		// Get evaluation limit
		evaluations_limit_ = Integer.parseInt(props.getProperty("Evaluations"));
		// Property keys depend on specific evaluation
		// E.g. double param = Double.parseDouble(props.getProperty("property_name"));
		boolean isMultimodal = Boolean.parseBoolean(props.getProperty("Multimodal"));
		boolean hasStructure = Boolean.parseBoolean(props.getProperty("Regular"));
		boolean isSeparable = Boolean.parseBoolean(props.getProperty("Separable"));

		// Do sth with property values, e.g. specify relevant settings of your algorithm
		if(isMultimodal){
			// Do sth
		}else{
			// Do sth else
		}

	}

	@SuppressWarnings("unused")
	public void run()
	{
		System.out.println("Run");

		setSeed(1); //TODO set a seed? or leave it for the parameter when running through console
		setEvaluation(sphere);

		int evals = 0;
		int popSize = INIT_POP_SIZE;
		double mutationRate = INIT_MUT_RATE;
		int nGenerations = 0;

		//set parameters for the three functions - setEvaluation
		if (false){
			//bentSigar
		} else if (false){
			//Schaffers
		} else{
			//Katsuura
		}

		// init population
		double[][] pop = init_population(popSize);
		//System.out.println(Arrays.deepToString(pop));
		System.out.println("Init first child: " + Arrays.toString(pop[0]));

		// calculate fitness
		double[] fit = calcFitness(pop, popSize);
		evals = evals + popSize; //add number of evaluations
		System.out.println("Init fitness list: " + Arrays.toString(round_array(fit)));

		double best_fit_in_gen = 54654; //for final fitness score, dummy value

		try{
			Boolean bool = true; //used to print some output during first generation
			while(evals<evaluations_limit_){

				// Select parents
				double[][] parents = parent_select(pop, NUMBER_OF_PARENTS); //TODO save parents' fitness in a variable, then these do not have to be calculated again. Saves evaluations     
				//if (bool) System.out.println("parent list: " + Arrays.deepToString(parents));
				if (bool) System.out.println("");

				// Apply crossover / mutation operators
				double[][] offspring = recombination(parents, NUMBER_OF_PARENTS, popSize);
				if (bool) System.out.println("Offspring first individual: " + Arrays.toString(round_array(offspring[0])));
				offspring = nonUniform_mutation(offspring, popSize);
				if (bool) System.out.println("Mutated first individual: " + Arrays.toString(round_array(offspring[0])));

				// Check fitness
				double[] offspring_fitness = calcFitness(offspring, popSize);
				evals = evals + popSize; //add to # evaluations
				if (bool) System.out.println("Offspring fitness list: " + Arrays.toString(round_array(offspring_fitness)));
				if (bool) System.out.println("");

				// Select survivors
				double[] parents_fitness = calcFitness(parents, NUMBER_OF_PARENTS); //TODO parents fitness should be known from previous round 
				pop = survivor_selection(offspring, offspring_fitness, parents, parents_fitness, 
						popSize, bool); //also update the population

				best_fit_in_gen = Math.max(getMax(offspring_fitness), getMax(parents_fitness));
				//print best fitness of generation and average of generation
				System.out.println(((double) Math.round(best_fit_in_gen*100))/100 + ";\t\t" + 
						getAverage(appendArray(offspring_fitness, parents_fitness))
				+ ";\t\t" + Arrays.toString(round_array(pop[0]))); //round by two decimals

				nGenerations++;
				if (nGenerations == DECREASE_POP_SIZE_AFTER){ //change some constanst after n Generations
					//popSize = (int) popSize/2;
				}
				bool = false; // only used for some print statements
			}
		} catch (NullPointerException e) {
			//show statistics of the run
			//can we retrieve the best fitness?

			System.out.println("-END STATISTICS-");
			//System.out.println("Final population: " + Arrays.deepToString(pop));
			//first child in population should be the best
			System.out.println("Best child in final population: " + Arrays.toString(pop[0]));
			System.out.println("Best fitness in final population: " + best_fit_in_gen);
			System.out.println(nGenerations + " # generations");
			
//			double[] a = {5,5,5,5,5,5,5,5,5,5};
////			can we get the score like this as well;
//			SphereEvaluation sphere2 = new SphereEvaluation();
//			setEvaluation(sphere2);
//			System.out.println((double) evaluation_.evaluate(a));
//			//        	System.out.println((double) evaluation_.evaluate(pop[1]));
//			//        	System.out.println((double) evaluation_.evaluate(pop[2]));
			//        	System.out.println((double) evaluation_.evaluate(pop[20]));

			//System.out.println(e); //print the exception: nullPointer
		}

	}

	public double[][] survivor_selection(double[][] lambda, double[] lambda_fitness, double[][] mu, double[] mu_fitness, int popSize, Boolean bool) {
		//somehow make a selection of survivor, for example:
		// 1) Fitness-Proportionate selection
		// 2) Tournament selection

		double[][] survivors = new double[popSize][lambda[0].length];
		double[][] pool = append2D(lambda, mu);
		double[] pool_fitness = appendArray(lambda_fitness, mu_fitness);

		int[] indices_sorted = getSortedFitnessIndices(pool_fitness);
		if (bool) System.out.println("Sorted fitness indices: " + Arrays.toString(indices_sorted));
		if (bool) System.out.print("Sorted fitness scores: ");
		if (bool) for (int i = 0; i < 10; i++) System.out.print(pool_fitness[indices_sorted[i]]+", "); //check if fitness values indeed sorted
		if (bool) System.out.print("\n\nBest fitness, Average Fitness, best child by generation: \n");

		survivors = selectTopN(pool, indices_sorted);
		return survivors;
	}

	public double[][] selectTopN(double[][] pool, int[] indices) {
		//simple takes the best X (population size) children from the pool. Best are specified with "indices"
		double[][] best = new double[pool.length][pool[0].length];

		for (int i = 0; i < pool.length; i++){
			best[i] = pool[indices[i]];
		}
		return best;
	}


	public int[] rouletteSelectN(double[][] pool, int N) { //From stackoverflow, but has been adjusted
		// Returns N selected index based on the weights
		double[] weight = getWeight(calcFitness(pool, pool.length));
		double weightSum = Arrays.stream(weight).sum();
		int[] indices = new int[N]; 

		for (int j=0; j<N; j++){

			double value = rnd_.nextDouble() * weightSum;	
			indices[j] = weight.length - 1; // when rounding errors occur, we return the last item's index, is overwritten if another is found
			for (int i=0; i<weight.length; i++) { // find the random value in roulette wheel
				value -= weight[i];		
				if(value < 0) {
					indices[j] = i;
					break;
				}
			}
		}
		return indices;
	}

	public int[] getSortedFitnessIndices(double[] values){
		//shitty function, but does the job

		//create hashmap for sorting
		HashMap<Integer, Double> map = new HashMap<Integer, Double>();

		//create map
		for (int i=0; i<values.length; i++){
			map.put(i, values[i]);
		}

		//apparently a hashmap is sorted easier
		Comparator<Integer> comparator = new ValueComparator<Integer, Double>(map);
		TreeMap<Integer, Double> result = new TreeMap<Integer, Double>(comparator);
		result.putAll(map);

		//get the keys from the hashmap
		int [] fitness_sorted_index = new int[values.length];
		int count = 0;
		for (int key : result.keySet()) {
			fitness_sorted_index[count] = key;
			count++;
		}
		return fitness_sorted_index;
	}

	public double[] createIndexArray(int min, int max){
		//inclusive start, exclusive end
		//TODO function no longer in use
		double[] result = new double[max-min];

		for (int i=min; i<max; i++){
			result[i] = i;
		}

		return result;
	}

	public double[] appendArray(double[] a, double[] b) {
		double[] result = new double[a.length + b.length];
		System.arraycopy(a, 0, result, 0, a.length);
		System.arraycopy(b, 0, result, a.length, b.length);
		return result;
	}

	public double[][] append2D(double[][] a, double[][] b) {
		double[][] result = new double[a.length + b.length][];
		System.arraycopy(a, 0, result, 0, a.length);
		System.arraycopy(b, 0, result, a.length, b.length);
		return result;
	}

	public double[][] nonUniform_mutation(double[][] offspring, int popSize) {
		double[][] offspring_mutated = new double[popSize][offspring[0].length];
		double mutation_stepSize = INIT_MUT_STEPSIZE;
		double mutationRate = INIT_MUT_RATE;

		for (int i=0; i<popSize; i++){
			for (int j=0; j<INIT_PHEN_SIZE; j++){
				//loop through one individual
				if (rnd_.nextDouble()<mutationRate){
					//mutate gene for some mutation rate, add random noise from Gaussian distribution
					offspring_mutated[i][j] = offspring[i][j] + (rnd_.nextGaussian()*mutation_stepSize);
				} else{
					offspring_mutated[i][j] = offspring[i][j];
				}
			}		
		}

		return offspring_mutated;
	}

	public double[][] recombination(double[][] parents, int nParents, int popSize) {
		//do we need the population, or just the parents?
		double[][] children = new double[popSize][parents[0].length];	

		for (int i=0; i<popSize; i++){
			//parents selected randomly
			double[] parentX = parents[rnd_.nextInt(nParents)];
			double[] parentY = parents[rnd_.nextInt(nParents)];;
			//children[i] = singlePointCrossover(parentX, parentY); //does single point crossover make sense for real values?
			children[i] = wholeArithmeticCrossover(parentX, parentY, 0.5); //can specify other fraction
		}

		return children;
	}

	public double[] wholeArithmeticCrossover(double[] X, double[] Y, double fraction){
		double[] child = new double[X.length];

		for (int i=0; i<X.length; i++){
			child[i] = fraction*X[i] + (1-fraction)*Y[i];
		}

		return child;
	}

	public double[] singlePointCrossover(double[] parentX, double[] parentY) {
		//not sure if this function makes sense for real values, but wholeArithmetic should probably be used anyways
		double[] child = new double[parentX.length];
		int crossOverPoint = rnd_.nextInt(parentX.length);

		for (int i=0; i<parentX.length; i++){
			if (i < crossOverPoint){
				child[i] = parentX[i];
			} else {
				child[i] = parentY[i];
			}
		}

		return child;
	}

	public double[][] parent_select(double[][] pool, int nParents){
		//used to do random selection, now roulette selection
		//random_select(pool, NUMBER_OF_PARENTS)
		
		double[][] parents = new double[nParents][pool[0].length];	
		//parents = random_select(pool, NUMBER_OF_PARENTS);
		//think that using weights might be more useful than this fitness value. Weights should be in [0,1]
		
		int[] parent_indices = rouletteSelectN(pool, nParents);
		for (int i=0; i<nParents; i++){
			parents[i] = pool[parent_indices[i]];
		}

		return parents;
	}
	
	public double[] getWeight(double[] fitness){
		double[] weights = new double[fitness.length];
		int count = 0;
		double totalWeight = 0;
		for (double f: fitness){ //get fitness between [-inf,-1], we should be able to retrieve the fitness from before
			weights[count] = f-11; //TODO hard value
			totalWeight += f-11;
			count++;
		}
		for (int i=0; i<fitness.length; i++){
			weights[i] = weights[i]/totalWeight; //low weight for very negative values, higher weight for values near -1
		}
		
		return weights;
	}

	public double[][] random_select(double[][] pop, int nParents){
		//random selection from population
		double[][] parents = new double[nParents][pop[0].length];	
		for (int i = 0; i<nParents; i++) parents[i] = pop[rnd_.nextInt(pop.length)]; //random parent selection, should be improved

		return parents;
	}

	public double[] calcFitness(double[][] pop, int size){
		//not sure if this is the fitness score meant, other score is described in the class
		int counter = 0;
		double[] result = new double[size];

		for (double[] ind: pop){
			result[counter] = (double) evaluation_.evaluate(ind);
			counter++;
		}

		return result;
	}

	public double[][] init_population(int size){ 
		//random number between [-high,high]
		int nElements = INIT_PHEN_SIZE;
		double[][] pop = new double[size][nElements];

		for (int i=0; i<size; i++){
			for (int j=0; j<nElements; j++){
				double randomNumber = rnd_.nextDouble() * 10 - 5; //search space [-5,5]
				//if (rnd_.nextBoolean()) randomNumber = -randomNumber;

				pop[i][j] = (double) Math.round(randomNumber*100)/100; //TODO remove round function
			}
		}
		return pop;
	}

	public double getMax(double[] array){
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < array.length; i++){
			if (array[i] > max) max = array[i];
		}
		return max;
	}

	public double getAverage(double[]  array){
		double sum = 0;
		for (double num: array) sum += num;
		return sum/array.length;
	}

	public double[] round_array(double[] array){
		for(int i=0; i<array.length; i++) array[i] = (double) Math.round(array[i]*100)/100;
		return array;
	}
	
	public class ValueComparator<K, V extends Comparable<V>> implements Comparator<K>{
		HashMap<K, V> map = new HashMap<K, V>();

		public ValueComparator(HashMap<K, V> map){
			this.map.putAll(map);
		}

		@Override
		public int compare(K s1, K s2) {
			return -map.get(s1).compareTo(map.get(s2)); //descending order
		}
	}

}
