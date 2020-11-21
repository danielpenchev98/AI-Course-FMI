#include <iostream>
#include <time.h>  
#include <random>
#include <chrono>
#include <algorithm>

std::mt19937_64 rng;
std::vector<std::pair<int, int>> cities;
std::vector<std::vector<double>> distances;
std::vector<int*> population;
std::vector<double> fitness;
int newGenerationSize;
const int tournamentSize = 10;
const double mutationProb = 0.08;
const double insertionMutProb = 1.0;
const double reverseMutProb = 0.4;
const double generateRandomProb = 0.2;

inline const std::pair<int*, int*> cyclicCrossover(const int* const parentA, const int* const parentB) {
	int* positionsA = new int[cities.size() + 1];
	int* firstChild = new int[cities.size()];
	int* secondChild = new int[cities.size()];

	for (int i = 0; i < cities.size(); i++) {
		firstChild[i] = -1;
		secondChild[i] = -1;
		positionsA[parentA[i]] = i;
	}

	int flag = true;
	for (int i = 0; i < cities.size(); i++) {
		if (firstChild[i] != -1) {
			continue;
		}

		int curr = i;
		while (firstChild[curr] == -1) {
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
	std::uniform_int_distribution<int> checkpointDistr(0, cities.size() - 1);
	int firstCheckpoint = checkpointDistr(rng);
	int secondCheckpoint = checkpointDistr(rng);
	int temp = secondCheckpoint;

	if (firstCheckpoint == secondCheckpoint) {
		return;
	}
	else if (firstCheckpoint > secondCheckpoint) {
		secondCheckpoint = firstCheckpoint;
		firstCheckpoint = temp;
	}

	temp = arr[secondCheckpoint];
	for (int i = secondCheckpoint - 1; i > firstCheckpoint; i--) {
		arr[i + 1] = arr[i];
	}
	arr[firstCheckpoint + 1] = temp;
}

inline void reverseSequenceMutation(int* const arr) {
	std::uniform_int_distribution<int> checkpointDistr(0, cities.size() - 1);
	int firstCheckpoint = checkpointDistr(rng);
	int secondCheckpoint = checkpointDistr(rng);
	int temp = secondCheckpoint;

	if (firstCheckpoint == secondCheckpoint) {
		return;
	}
	else if (firstCheckpoint > secondCheckpoint) {
		secondCheckpoint = firstCheckpoint;
		firstCheckpoint = temp;
	}

	int areaSize = secondCheckpoint - firstCheckpoint + 1;
	int center = areaSize / 2;
	for (int i = 0; i < center; i++) {
		temp = arr[i + firstCheckpoint];
		arr[i + firstCheckpoint] = arr[secondCheckpoint - i];
		arr[secondCheckpoint - i] = temp;
	}
}

inline void mutate(int* const arr) {
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	double prob = dis(rng);
	if (prob > mutationProb) {
		return;
	}

	double picker = dis(rng);
	if (picker <= generateRandomProb) {
		std::shuffle(arr, arr + cities.size(), rng);
	}
	else if (picker <= reverseMutProb) {
		reverseSequenceMutation(arr);
	}
	else {
		insertMutation(arr);
	}

}

inline int getTournamentWinner(const std::vector<int>& competitors, const std::vector<double>& allCompetitorfitness) {
	int winnerIdx = competitors[0];
	double bestFitness = allCompetitorfitness[winnerIdx];
	for (int compIdx : competitors) {
		if (allCompetitorfitness[compIdx] < bestFitness) {
			bestFitness = allCompetitorfitness[compIdx];
			winnerIdx = compIdx;
		}
	}
	return winnerIdx;
}

//using stochastic universal sampling
inline std::vector<int> rouletteWheenSelection() {
	int winnerCount = 0.3 * population.size();
	double totalFitness = 0;
	for (int i = 0; i < population.size(); i++) {
		totalFitness += fitness[i];
	}

	double* commulativeFitness = new double[population.size()];

	commulativeFitness[0] = fitness[0] / totalFitness;
	for (int i = 1; i < population.size(); i++) {
		commulativeFitness[i] = commulativeFitness[i - 1] + (fitness[i] / totalFitness);
	}

	int curr = 0;
	std::uniform_real_distribution<> dis(0.0, 1.0 / winnerCount);
	double r = dis(rng);
	int i = 0;

	std::vector<int> winners;
	while (curr < winnerCount) {
		while (r <= commulativeFitness[i]) {
			winners.push_back(i);
			curr++;
			r += 1.0 / winnerCount;
		}
		i++;
	}

	delete[] commulativeFitness;
	return winners;
}

inline double calculateFitness(const int* const individual) {
	double fitness = 0;
	for (int i = 1; i < cities.size(); i++) {
		fitness += distances[individual[i - 1]][individual[i]];
	}
	return fitness;
}

inline const std::vector<int> roundRobinTournament(const std::vector<int*>& gladiators, const std::vector<double>& gladiatorFitness, int tournamentsNumbers, int winnerNumbers) {
	std::vector<std::pair<int, int>> results(gladiators.size());
	for (int i = 0; i < gladiators.size(); i++) {
		results[i].first = i;
		results[i].second = 0;
	}

	std::uniform_int_distribution<int> distr(0, gladiators.size() - 1);
	std::vector<int> tournamentCompetitors(tournamentSize);
	
	for (int i = 0; i < tournamentsNumbers; i++) {
		for (int j = 0; j < tournamentSize; j++) {
			int competitor = distr(rng);
			tournamentCompetitors[j] = competitor;
		}
		int winner = getTournamentWinner(tournamentCompetitors, gladiatorFitness);
		results[winner].second++;
	}

	//sort by their wins in descending order and also if have same win counter by fitness
	std::sort(results.begin(), results.end(), [&](const std::pair<int, int>& l, const std::pair<int, int>& r) {
		if (l.second == r.second) {
			return gladiatorFitness[l.first] < gladiatorFitness[r.first];
		}
		return l.second > r.second; });

	std::vector<int> winners(winnerNumbers);
	for (int i = 0; i < winnerNumbers; i++) {
		winners[winnerNumbers - i - 1] = results[i].first;
	}

	//sort by descending fitness
	std::sort(winners.begin(), winners.end(), [&](const int l, const int r) {
		return gladiatorFitness[l] > gladiatorFitness[r]; });

	return winners;
}

//needs big population + no duplicates - 10% elitism + 90% and newgeneration round robin
inline void updatePopulation(const std::vector<int*>& newGeneration) {
	int elitism = 0.1 * population.size();

	int competitiorNumbers = population.size() - elitism + newGeneration.size();

	std::vector<int*> competitors(competitiorNumbers);
	std::vector<double> competitorFitness(competitiorNumbers);

	for (int i = 0; i < competitiorNumbers; i++) {
		if (i < population.size() - elitism) {
			competitors[i] = population[i];
			competitorFitness[i] = fitness[i];
		}
		else
		{
			competitors[i] = newGeneration[i - population.size() + elitism];
			competitorFitness[i] = calculateFitness(competitors[i]);
		}
	}

	const std::vector<int>& winners = roundRobinTournament(competitors, competitorFitness, population.size(), population.size() - elitism);

	for (int i = 0; i < population.size() - elitism; i++) {
		population[i] = competitors[winners[i]];
		fitness[i] = competitorFitness[winners[i]];
		competitors[winners[i]] = nullptr;
	}

	//for small population like 1000 and also its almost sorted
	for (int i = population.size() - elitism; i < population.size(); i++) {
		int* temp = population[i];
		double fit = fitness[i];
		int j = i - 1;
		for (; j >= 0 && fitness[j] < fit; j--) {
			population[j + 1] = population[j];
			fitness[j + 1] = fitness[j];
		}
		fitness[j + 1] = fit;
		population[j + 1] = temp;
	}

	for (int i = 0; i < population.size() - elitism + newGeneration.size(); i++) {
		if (competitors[i] == nullptr) continue;
		delete[] competitors[i];
	}
}

inline void showPopulationStatistics() {
	double mean = 0;
	double bestFitness = DBL_MAX;
	double worstFitness = 0;
	for (int i = 0; i < population.size(); i++) {
		mean += fitness[i] / population.size();
		if (bestFitness > fitness[i]) {
			bestFitness = fitness[i];
		}

		if (worstFitness < fitness[i]) {
			worstFitness = fitness[i];
		}
	}

	std::cout << "Best :" << bestFitness << " Worst :" << worstFitness << " Mean :" << mean << std::endl;
}

inline const std::vector<int*> getNewGeneration(const std::vector<int>& winners) {
	std::vector<int*> newGeneration;
	for (int i = 0; i < newGenerationSize; i++) {
		std::uniform_int_distribution<int> distr(0, winners.size() - 1);
		int firstParent = winners[distr(rng)];
		int secondParent = winners[distr(rng)];

		//crossover
		std::pair<int*, int*> children = cyclicCrossover(population[firstParent], population[secondParent]);

		//mutation
		mutate(children.first);
		mutate(children.second);

		newGeneration.push_back(children.first);
		newGeneration.push_back(children.second);
	}

	for (int i = 0; i < newGenerationSize; i++) {
		if (newGeneration[i] == nullptr) {
			exit(1);
		}
	}

	return newGeneration;
}

void initCities() {
	int cityNumbers = 0;
	std::cin >> cityNumbers;
	cities.resize(cityNumbers);
	distances.resize(cities.size());

	int x, y;
	for (int i = 0; i < cities.size(); i++) {
		std::cin >> x;
		cities[i].first = x;
		std::cin >> y;
		cities[i].second = y;
		distances[i].resize(cities.size());
	}

	for (int i = 0; i < cities.size(); i++) {
		for (int j = 0; j < cities.size(); j++) {
			distances[i][j] = sqrt((cities[i].first - cities[j].first) * (cities[i].first - cities[j].first) +
				(cities[i].second - cities[j].second) * (cities[i].second - cities[j].second));
		}
	}
}

void initPopulation() {
	int populationSize;
	std::cout << "Choose population number :";
	std::cin >> populationSize;

	//Set children to be 20% of the population
	newGenerationSize = 0.3 * populationSize;

	int* arr = new int[cities.size()];
	for (int i = 0; i < cities.size(); i++) {
		arr[i] = i;
	}

	population.resize(populationSize);
	fitness.resize(populationSize);
	for (int i = 0; i < population.size(); i++) {
		population[i] = new int[cities.size()];
		std::shuffle(arr, arr + cities.size(), rng);
		for (int j = 0; j < cities.size(); j++) {
			population[i][j] = arr[j];
		}
	}

	std::sort(population.begin(), population.end(), [](const int* const l, const int* const r) {
		return calculateFitness(l) > calculateFitness(r); });

	for (int i = 0; i < population.size(); i++) {
		fitness[i] = 0;
		for (int j = 1; j < cities.size(); j++) {
			fitness[i] += distances[population[i][j - 1]][population[i][j]];
		}
	}
}

int main() {
	rng.seed(std::random_device{}());
	initCities();
	initPopulation();

	int thresholdGenerations = 400;
	int currGeneration = 0;
	while (currGeneration <= thresholdGenerations) {
		//std::cout << "Generation :" << currGeneration << " ";
		//Selection step
		const std::vector<int>& winners = rouletteWheenSelection();//tournamentSelection();

		//Breeding step
		const std::vector<int*>& newGeneration = getNewGeneration(winners);

		//Survival step
		updatePopulation(newGeneration);
		//showPopulationStatistics();

		currGeneration++;
	}

	int bestIndx = 0;
	double bestFitness = DBL_MAX;
	for (int i = 0; i < population.size(); i++) {
		if (fitness[i] < bestFitness) {
			bestFitness = fitness[i];
			bestIndx = i;
		}
	}

	std::cout << bestFitness << " ";

	for (int i = 0; i < cities.size(); i++) {
		std::cout << population[bestIndx][i] << " ";
	}

	for (int i = 0; i < population.size(); i++) {
		delete[] population[i];
	}

	return 0;
}