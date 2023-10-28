#pragma once

#include <cstdlib>
#include <random>

inline float random01() 
{
    return std::rand() * 1.F / RAND_MAX;
}

inline float random(float a, float b)
{
    return a + (b - a) * random01();
}

inline void clamp(float& t, float mn, float mx)
{
    t = std::min(std::max(t, mn), mx);
}