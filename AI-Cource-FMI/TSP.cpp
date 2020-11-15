#include <iostream>
#include <time.h>  
#include <random>
#include <chrono>
#include <algorithm>

std::mt19937_64 rng;

std::pair<int, int>* verteces;
double** distances;
int** population;
int populationSize;
double* fitness;
int cityNumbers;
const int tournamentSize = 10;
int generationSize = 25000;
const double mutationProbability = 0.05;

inline const std::pair<int*,int*> cyclicCrossover(int* parentA, int* parentB) {
	int* positionsA = new int[cityNumbers+1];
	int* firstChild = new int[cityNumbers];
	int* secondChild = new int[cityNumbers];


	
	for (int i = 0; i < cityNumbers; i++) {
		firstChild[i] = -1;
		secondChild[i] = -1;
		positionsA[parentA[i]] = i;
	}

	int flag = true;
	for (int i = 0; i < cityNumbers; i++) {
		if (firstChild[i] != -1) {
			continue;
		}

		int curr = i;
		while (firstChild[curr]==-1) {
			if (flag) {
				firstChild[curr] = parentA[curr];
				secondChild[curr] = parentB[curr];
			}
			else {
				firstChild[curr] = parentB[curr];
				secondChild[curr] = parentA[curr];
			}
			curr = positionsA[parentB[curr]];
		}

		flag = !flag;
	}

	

	delete[] positionsA;

	return std::pair<int*, int*>(firstChild, secondChild);
}

inline void insertMutation(int* const arr) {
	std::uniform_real_distribution<> dis(0.0, 1.0);
	double prob = dis(rng);
	if (prob > mutationProbability) {
		return;
	}

	int firstCheckpoint = rng() % cityNumbers;
	int secondCheckpoint = rng() % cityNumbers;
	int temp = secondCheckpoint;

	if (firstCheckpoint == secondCheckpoint) {
		return;
	} else if (firstCheckpoint > secondCheckpoint) {
		secondCheckpoint = firstCheckpoint;
		firstCheckpoint = temp;
	}

	temp = arr[secondCheckpoint];
	for (int i = secondCheckpoint -1; i > firstCheckpoint; i--) {
		arr[i+1] = arr[i];
	}
	arr[firstCheckpoint + 1] = temp;
}

inline int getTournamentWinner(const std::vector<int>& competitors) {
	int winnerIdx = competitors[0];
	double bestFitness = fitness[0];
	for (int compIdx : competitors) {
		if (fitness[compIdx] < bestFitness) {
			bestFitness = fitness[compIdx];
			winnerIdx = compIdx;
		}
	}
	return winnerIdx;
}

//for the surviving mechanism use maybe Crowding

inline std::vector<int> tournamentSelection() {
	std::vector<int> winners(25000);
	std::vector<int> tournamentCompetitors(tournamentSize);
	for (int i = 0; i < 25000; i++) {
		for (int j = 0; j < tournamentSize; j++) {
			//should a competitor play in the same tournament twice
			int competitor = rng() % populationSize;
			tournamentCompetitors[j] = competitor;
		}
		winners[i] = getTournamentWinner(tournamentCompetitors);
	}
	return winners;
}

//using stochastic universal sampling
/*inline std::vector<int> rouletteWheenSelection() {
	double totalFitness = 0;
	for (int i = 0; i < populationSize; i++) {
		totalFitness += fitness[i];
	}

	double* commulativeFitness = new double[populationSize];

	commulativeFitness[0] = fitness[0] / totalFitness;
	for (int i = 1; i < populationSize; i++) {
		commulativeFitness[i] = commulativeFitness[i - 1] + (fitness[i] / totalFitness);
	}

	std::cout << commulativeFitness[populationSize - 1] << std::endl;

	int curr = 0;
	std::uniform_real_distribution<> dis(0.0, 1.0/generationSize);
	double r = dis(rng);
	int i = 0;

	std::vector<int> winners;
	while (curr < generationSize) {
		while (r <= commulativeFitness[i]) {
			winners.push_back(i);
			curr++;
			r += 1.0 / generationSize;
		}
		i++;
	}
	return winners;
}*/

inline double calculateFitness(const int* const individual) {
	double fitness = 0;
	for (int i = 1; i < cityNumbers+1; i++) {
		fitness += distances[individual[i - 1]][individual[i]];
	}
	return fitness;
}

//needs big population + no duplicates - Replace worst(GENTITOR)
inline void updatePopulation(const std::vector<int*> newGeneration) {
	int** mergedPopulation = new int*[populationSize + newGeneration.size()];


	double worstFitness = 0;
	int* bestIndividual = nullptr;
	double bestFitness = DBL_MAX;
	for (int i = 0; i < populationSize + newGeneration.size(); i++) {
		if (i < populationSize) {
			mergedPopulation[i] = population[i];
		}
		else 
		{
			mergedPopulation[i] = newGeneration[i - populationSize];
		}

		double currFitness = calculateFitness(mergedPopulation[i]);
		if (bestFitness > currFitness) {
			bestFitness = currFitness;
			bestIndividual = mergedPopulation[i];
		}

		if (worstFitness < currFitness) {
			worstFitness = currFitness;
		}
	}

	std::sort(mergedPopulation, mergedPopulation + populationSize+newGeneration.size(), [](const int* const l,const int* const r) {
		return calculateFitness(l) < calculateFitness(r); });

	std::cout << "BestFitness is :" << bestFitness <<" ";
	std::cout << "Worst Fitness is :" << worstFitness << " ";
	double mean = 0;
	for (int i = 0; i < populationSize; i++) {
		population[i] = mergedPopulation[i];
		fitness[i] = calculateFitness(population[i]);
		mean += fitness[i] / populationSize;
	}

	std::cout << "Mean :" << mean << std::endl;

	for (int i = 0; i < newGeneration.size(); i++) {
		delete[] mergedPopulation[i+populationSize];
	}
	delete[] mergedPopulation;
}

inline std::vector<int*> getNewGeneration(const std::vector<int>& winners) {
	std::vector<int*> newGeneration;
	for (int i = 0; i < generationSize; i++) {
		int firstParent = winners[rng() % winners.size()];
		int secondParent = winners[rng() % winners.size()];

		//crossover
		std::pair<int*, int*> children = cyclicCrossover(population[firstParent], population[secondParent]);

		//mutation
		insertMutation(children.first);
		insertMutation(children.second);

		newGeneration.push_back(children.first);
		newGeneration.push_back(children.second);
	}

	return newGeneration;
}

int main() {
	std::cin >> cityNumbers;

	verteces = new std::pair<int, int>[cityNumbers];
	rng.seed(std::random_device{}());

	int x, y;
	for (int i = 0; i < cityNumbers; i++) {
		std::cin >> x;
		std::cin >> y;
		verteces[i] = std::pair<int, int>(x, y);
	}

	
	distances = new double* [cityNumbers];
	for (int i = 0; i < cityNumbers; i++) {
		distances[i] = new double[cityNumbers];
		for (int j = 0; j < cityNumbers; j++) {
			distances[i][j] = sqrt((verteces[i].first - verteces[j].first) * (verteces[i].first - verteces[j].first) +
				(verteces[i].second - verteces[j].second) * (verteces[i].second - verteces[j].second));
		}
	}

	std::cout << "Choose population number :";
	std::cin >> populationSize;
	
	int* arr = new int[cityNumbers];
	for (int i = 0; i < cityNumbers; i++) {
		arr[i] = i;
	}

	/*arr[0] = 40;
	arr[1] = 15;
	arr[2] = 21;
	arr[3] = 0;
	arr[4] = 7;
	arr[5] = 8;
	arr[6] = 37;
	arr[7] = 30; arr[8] = 43; arr[9] = 17; arr[10] = 6; arr[11] = 27;
	arr[12] = 5; arr[13] = 36; arr[14] = 18; arr[15] = 26; arr[16] = 16; arr[17] = 42; arr[18] = 29; arr[19] = 35; arr[20] = 45; arr[21] = 32; arr[22] = 19;arr[23] = 10; arr[24] = 22;
	arr[25] = 12; arr[26] = 24; arr[27] = 13; arr[28] = 33; arr[29] = 2; arr[30] = 39; arr[31] = 14; arr[32] = 11; arr[33] = 46; arr[34] = 20; arr[35] = 31; arr[36] = 38; arr[37] = 47; arr[38] = 4;
	arr[39] = 28; arr[40] = 1; arr[41] = 3; arr[42] = 25; arr[43] = 41; arr[44] = 23; arr[45] = 9; arr[46] = 34; arr[47] = 44; arr[48] = 40;*/

	std::shuffle(arr, arr + cityNumbers, rng);
	std::shuffle(arr, arr + cityNumbers, rng);
	std::shuffle(arr, arr + cityNumbers, rng);
	std::shuffle(arr, arr + cityNumbers, rng);
	population = new int* [populationSize];
	fitness = new double[populationSize];
	for (int i = 0; i < populationSize; i++) {
		population[i] = new int[cityNumbers];
		std::shuffle(arr, arr + cityNumbers, rng);
		for (int j = 0; j < cityNumbers; j++) {
			population[i][j] = arr[j];
		}

		fitness[i] = 0;
		for (int j = 1; j < cityNumbers; j++) {
			fitness[i] += distances[population[i][j - 1]][population[i][j]];
		}
	}
	int thresholdGenerations = 5000;  
	int currGeneration = 0;
	while (currGeneration<=thresholdGenerations) {
		std::cout << "Generation :" << currGeneration <<" ";
		//Selection step
		const std::vector<int>& winners = tournamentSelection();
		
		//Breeding step
		const std::vector<int*>& newGeneration = getNewGeneration(winners);

		//Survival step
		updatePopulation(newGeneration);

		currGeneration++;
	}
	
	int bestIndx = 0;
	double bestFitness = DBL_MAX;
	for (int i = 0; i < populationSize; i++) {
		if (fitness[i] < bestFitness) {
			bestFitness = fitness[i];
			bestIndx = i;
		}
	}
	
	std::cout << bestFitness<<" ";

	for (int i = 0; i < cityNumbers; i++) {
		std::cout << population[bestIndx][i]<<" ";
	}
	
	return 0;
}