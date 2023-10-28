#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>

#include "utils.hpp"

template <typename Trait>
struct Problem
{
        struct Solution
        {
            Solution(typename Trait::Type&& value)
                : value(std::move(value))
                , fitness(Trait::fitness(this->value))
            {}
            Solution& operator = (Solution&& sol)
            {
                value = std::move(sol.value);
                fitness = sol.fitness;
            }
            typename Trait::Type value;
            float fitness;
            void mutate()
            {
                Trait::mutate(value);
                fitness = Trait::fitness(value);
            }
        };
        using Type = std::unique_ptr<Solution>;

        Problem(std::size_t pop_size = 150, std::size_t cx_count = 50, std::size_t gen = 100, float mut_rate = 0.05F, bool verbose = true)
            : m_population_size(pop_size)
            , m_crossover_count(cx_count)
            , m_generation_count(gen)
            , m_mutation_rate(mut_rate)
            , m_verbose(verbose)
        {}

        void run()
        {
            for (int i = 0; i < m_population_size; ++i)
            {
                m_population.emplace_back(std::make_unique<Solution>(Trait::create()));
            }
            rank_population();

            int current_gen = 0;
            while (current_gen < m_generation_count)
            {
                current_gen++;
                crossover();
                select();
                if (m_verbose)
                    std::cout << "Generation " << current_gen << ", Best = " << m_population[0]->fitness << '\n';
            }
            std::cout << "Best fitness found: " << m_population[0]->fitness << '\n';
            Trait::print(m_population[0]->value);
        }
        void crossover()
        {
            for (int i = 0; i < m_crossover_count; ++i)
                for (int j = 0; j < i; ++j)
                {
                    auto& pa = m_population[i]->value;
                    auto& pb = m_population[j]->value;
                    m_population.emplace_back(std::make_unique<Solution>(Trait::crossover(pa, pb)));
                    auto& pc = *m_population.back();
                    if (random01() < m_mutation_rate)
                        pc.mutate();
                }
        }
        void select()
        {
            rank_population();
            m_population.erase(m_population.begin() + m_population_size, m_population.end());
        }
    private:
        void rank_population()
        {
            std::sort(m_population.begin(), m_population.end(), [](const Type& sml, const Type& lgr) {
                return Trait::fitness(sml->value) < Trait::fitness(lgr->value);
            });
        }
        const std::size_t m_population_size;
        const std::size_t m_crossover_count;
        const std::size_t m_generation_count;
        const float m_mutation_rate;
        const bool m_verbose{ true };
        std::vector<Type> m_population;
};
