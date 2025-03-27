#Lana Musaffer || 1210455
#Mayar Masalmeh || 1211246

import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import matplotlib.colors as mcolors
import pandas as pd

SCALING_FACTOR = 100   # Define the scaling factor

# Function to parse job details from strings
def parse_jobs(job_strings: List[str]) -> Tuple[List[List[Tuple[str, int]]], List[str]]:
    jobs = []
    all_machines = []
    for job_str in job_strings:
        operations = job_str.split("->")
        job = []
        for operation in operations:
            machine, time = operation.strip().split("[")
            time = int(time[:-1])  # Remove the closing bracket and convert to int
            job.append((machine, time))
            if machine not in all_machines:
                all_machines.append(machine)
        jobs.append(job)
    return jobs, all_machines


# Function to calculate makespan given a list of jobs and machines
def calculate_makespan(jobs: List[List[Tuple[str, int]]], all_machines: List[str]) -> int:
    machine_end_times = {machine: 0 for machine in all_machines}
    job_end_times = [0] * len(jobs)
    
    for job_index, job in enumerate(jobs):
        current_time = 0
        for machine, time in job:
            start_time = max(current_time, machine_end_times[machine])
            end_time = start_time + time
            machine_end_times[machine] = end_time
            current_time = end_time
        job_end_times[job_index] = current_time

    makespan = max(job_end_times)
    return makespan


# Function to evaluate fitness given job strings
def evaluate_fitness(job_strings: List[str]) -> float:
    # Parse job details
    jobs, all_machines = parse_jobs(job_strings)

    # Calculate makespan
    makespan = calculate_makespan(jobs, all_machines)

    # Scale the inverse makespan to fitness values
    fitness_value = SCALING_FACTOR / makespan

    return fitness_value


# Function to generate an ordered schedule from a list of jobs
def generate_ordered_schedule(jobs: List[List[Tuple[str, int]]]) -> List[Tuple[int, str, int]]:
    all_operations = []
    for job_index, job in enumerate(jobs):
        for operation_index, (machine, time) in enumerate(job):
            all_operations.append((job_index, machine, time, operation_index))
    # Sort the operations based on their position within the job
    all_operations.sort(key=lambda x: x[3])
    # Remove the operation index from the tuples
    ordered_schedule = [(job_index, machine, time) for job_index, machine, time, _ in all_operations]
    return ordered_schedule


# Function to create the initial population of schedules
def create_initial_population(jobs: List[List[Tuple[str, int]]], population_size: int) -> List[List[Tuple[int, str, int]]]:
    population = []
    for _ in range(population_size):
        schedule = generate_ordered_schedule(jobs)
        # Ensure we shuffle only the jobs, not the operations within each job
        job_indices = list(range(len(jobs)))
        random.shuffle(job_indices)
        shuffled_schedule = []
        for job_index in job_indices:
            shuffled_schedule.extend([(job_index, machine, time) for machine, time in jobs[job_index]])
        population.append(shuffled_schedule)
    return population


# Function to plot a Gantt chart for a given schedule
def plot_gantt_chart(schedule: List[Tuple[int, str, int]], jobs: List[List[Tuple[str, int]]], all_machines: List[str], title: str):
    machine_timeline = {machine: [] for machine in all_machines}
    job_end_times = [0] * len(jobs)
    job_starts = [0] * len(jobs)

    # Create a color map for jobs
    colors = list(mcolors.TABLEAU_COLORS.values())
    job_colors = {i: colors[i % len(colors)] for i in range(len(jobs))}

    # Place the operations in the correct order for each job
    for job_index, machine, time in schedule:
        start_time = max(job_starts[job_index], machine_timeline[machine][-1][1] if machine_timeline[machine] else 0)
        end_time = start_time + time
        machine_timeline[machine].append((start_time, end_time, job_index))
        job_starts[job_index] = end_time

    fig, ax = plt.subplots()

    for machine in all_machines:
        if machine in machine_timeline:
            tasks = machine_timeline[machine]
            for task in tasks:
                start_time, end_time, job_index = task
                ax.barh(machine, end_time - start_time, left=start_time, edgecolor='black', color=job_colors[job_index], label=f'Job {job_index+1}' if machine == all_machines[0] else "")
                ax.text((start_time + end_time) / 2, machine, f'J{job_index+1}', color='black', ha='center', va='center')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Time Units')
    ax.set_ylabel('Machines')
    ax.set_title(title)
    plt.show()


# Function to read job details from a CSV file
def get_job_details_from_csv(file_path: str) -> Tuple[int, List[str]]:
    try:
        df = pd.read_csv(file_path)
        jobs = df['Operations'].tolist()
        num_of_machines = len(set(machine for job in jobs for machine in job if machine.isalpha()))
        return num_of_machines, jobs
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return 0, []
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, []


# Mutation operator: Inversion mutation
def inversion_mutation(chromosome: List[Tuple[int, str, int]]) -> List[Tuple[int, str, int]]:
    # Select a job randomly
    job_index = random.choice(list(set(job for job, _, _ in chromosome)))
    job_positions = [i for i, (j, _, _) in enumerate(chromosome) if j == job_index]
    if len(job_positions) > 1:
        pos1, pos2 = sorted(random.sample(job_positions, 2))
        # Invert the substring between the selected positions
        chromosome[pos1:pos2+1] = reversed(chromosome[pos1:pos2+1])
    return chromosome


# Mutation operator: Insertion mutation
def insertion_mutation(chromosome: List[Tuple[int, str, int]]) -> List[Tuple[int, str, int]]:
    # Select a job randomly
    job_index = random.choice(list(set(job for job, _, _ in chromosome)))
    job_positions = [i for i, (j, _, _) in enumerate(chromosome) if j == job_index]
    if len(job_positions) > 1:
        pos1, pos2 = random.sample(job_positions, 2)
        # Insert the element at pos2 before the element at pos1
        element = chromosome.pop(pos2)
        chromosome.insert(pos1, element)
    return chromosome


# Function to apply mutation to the population
def apply_mutation(population: List[List[Tuple[int, str, int]]], mutation_rate: float) -> List[List[Tuple[int, str, int]]]:
    mutated_population = []
    for chromosome in population:
        # Apply mutation with a certain probability
        if random.random() < mutation_rate:
            # Choose a mutation operator randomly
            mutation_operator = random.choice([inversion_mutation, insertion_mutation])
            # Apply the selected mutation operator
            mutated_chromosome = mutation_operator(chromosome.copy())
            mutated_population.append(mutated_chromosome)
        else:
            mutated_population.append(chromosome)
    return mutated_population


# Crossover operator: Partially Mapped Crossover (PMX)
def crossover(parent1: List[Tuple[int, str, int]], parent2: List[Tuple[int, str, int]], num_jobs: int) -> List[Tuple[int, str, int]]:

    # Choose a crossover point randomly
    crossover_point = random.randint(1, len(parent1) - 2)

    # Create a child by copying the segment from parent1 up to the crossover point
    child = parent1[:crossover_point]

    # Fill in the remaining positions in the child with elements from parent2
    remaining_jobs = [job for job in parent2 if job[0] not in [op[0] for op in child]]
    child.extend(remaining_jobs)

    return child


# Function to apply crossover to the population
def apply_crossover(population: List[List[Tuple[int, str, int]]], crossover_rate: float, num_jobs: int) -> List[List[Tuple[int, str, int]]]:
    new_population = []
    while len(new_population) < len(population):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        if random.random() < crossover_rate:
            # Apply crossover
            child = crossover(parent1, parent2, num_jobs)
            new_population.append(child)
        else:
            new_population.append(parent1)
            if len(new_population) < len(population):
                new_population.append(parent2)
    return new_population


# Function to evaluate fitness for the population
def evaluate_population(population: List[List[Tuple[int, str, int]]], jobs: List[List[Tuple[str, int]]], all_machines: List[str]) -> List[float]:
    fitness_values = []
    for schedule in population:
        reordered_jobs = [[] for _ in range(len(jobs))]
        for job_index, machine, time in schedule:
            reordered_jobs[job_index].append((machine, time))
        makespan = calculate_makespan(reordered_jobs, all_machines)
        fitness_value = SCALING_FACTOR / makespan
        fitness_values.append(fitness_value)
    return fitness_values


# Function to validate a chromosome
def validate_chromosome(chromosome: List[Tuple[int, str, int]], num_jobs: int, jobs: List[List[Tuple[str, int]]]) -> bool:
    job_operation_count = {i: 0 for i in range(num_jobs)}
    for job_index, _, _ in chromosome:
        job_operation_count[job_index] += 1

    for job_index, job in enumerate(jobs):
        if job_operation_count[job_index] != len(job):
            return False
    return True


# Function for proportional selection of parents
def proportional_selection(population: List[List[Tuple[int, str, int]]], fitness_values: List[float], num_parents: int) -> List[List[Tuple[int, str, int]]]:
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        # If total fitness is zero, assign equal probabilities to all individuals
        selection_probabilities = [1 / len(population)] * len(population)
    else:
        selection_probabilities = [fitness / total_fitness for fitness in fitness_values]

    # Select parents based on their fitness values
    selected_indices = random.choices(range(len(population)), weights=selection_probabilities, k=num_parents)
    selected_individuals = [population[i] for i in selected_indices]

    return selected_individuals


# Function to ensure a valid population
def ensure_valid_population(population: List[List[Tuple[int, str, int]]], num_jobs: int, jobs: List[List[Tuple[str, int]]]) -> List[List[Tuple[int, str, int]]]:
    valid_population = []
    for chromosome in population:
        if validate_chromosome(chromosome, num_jobs, jobs):
            valid_population.append(chromosome)
    return valid_population


# Main function
def main():
    file_path = 'jobs.csv'  # Path to the CSV file
    num_of_machines, job_strings = get_job_details_from_csv(file_path)
    if num_of_machines == 0 or not job_strings:
        print("Failed to read job details from CSV.")
        return

    population_size = 10  # Define the size of the initial population
    mutation_rate = 0.1  # Mutation rate (adjust as needed)
    crossover_rate = 0.8  # Crossover rate (adjust as needed)

    # Parse job details
    jobs, all_machines = parse_jobs(job_strings)

    # Create the initial population
    initial_population = create_initial_population(jobs, population_size)

    # Ensure initial population is valid
    valid_population = ensure_valid_population(initial_population, len(jobs), jobs)

    # Apply crossover to the initial population
    crossover_population = apply_crossover(valid_population, crossover_rate, len(jobs))

    # Apply mutation to the crossover population
    mutated_population = apply_mutation(crossover_population, mutation_rate)

    # Ensure mutated population is valid
    final_population = ensure_valid_population(mutated_population, len(jobs), jobs)

    # Track the best solution found so far
    best_solution = None
    best_makespan = float('inf')  # Initialize best makespan with infinity

    # Evaluate fitness for the final population and find the best makespan
    for i, schedule in enumerate(final_population):
        print(f"Chromosome {i+1}: {schedule}")

        reordered_jobs = [[] for _ in range(len(jobs))]
        for job_index, machine, time in schedule:
            reordered_jobs[job_index].append((machine, time))
        makespan = calculate_makespan(reordered_jobs, all_machines)

        print(f"Makespan for Chromosome {i+1}: {makespan} time units")

        # Plot Gantt chart for the chromosome
        plot_gantt_chart(schedule, jobs, all_machines, f"Gantt Chart for Chromosome {i+1}")

        # Update the best solution if a new best solution is found
        if makespan < best_makespan:
            best_solution = schedule
            best_makespan = makespan

    # Print the best solution found
    print("\nBest Solution:")
    print(best_solution)
    print(f"Best Makespan: {best_makespan} time units")

    # Evaluate fitness for the given input
    fitness = SCALING_FACTOR / best_makespan if best_makespan != float('inf') else 0
    print("Fitness value:", fitness)

    return best_solution

if _name_ == "_main_":
    best_chromosome = main()