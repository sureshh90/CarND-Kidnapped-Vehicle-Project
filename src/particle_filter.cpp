/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * NOTE: Sets the number of particles. Initializes all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * NOTE: Adds random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   */
  num_particles = 50; // Set the number of particles

  // Create normal (Gaussian) distributions for x,  y and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  std::default_random_engine gen; // initialise a random distribution engine

  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;

    // Sample from these normal distributions like this:
    //   e.g. sample_x = dist_x(gen);
    //   where "gen" is the random engine initialized earlier.
    p.id = i + 1;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;

    particles.push_back(p);
  }

  is_initialized = true; // Set the initialised flag
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * NOTE: Adds measurements to each particle with a random Gaussian noise.
   * NOTE: When adding noise the functions std::normal_distribution 
   *   and std::default_random_engine are useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Create normal (Gaussian) distributions for x,  y and theta
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  std::default_random_engine gen; // initialise a random distribution engine

  std::for_each(particles.begin(), particles.end(), [&](Particle &current_particle) {
    double new_x = 0;
    double new_y = 0;
    double new_theta = 0;

    if (yaw_rate == 0)
    {
      new_x = current_particle.x + (velocity * delta_t * cos(current_particle.theta));
      new_y = current_particle.y + (velocity * delta_t * sin(current_particle.theta));
      new_theta = current_particle.theta;
    }
    else
    {
      new_x = current_particle.x + ((velocity / yaw_rate) * (sin(current_particle.theta + (yaw_rate * delta_t)) - sin(current_particle.theta)));
      new_y = current_particle.y + ((velocity / yaw_rate) * (cos(current_particle.theta) - cos(current_particle.theta + (yaw_rate * delta_t))));
      new_theta = current_particle.theta + (yaw_rate * delta_t);
    }

    // Sample from these normal distributions
    auto noise_x = dist_x(gen);
    auto noise_y = dist_y(gen);
    auto noise_theta = dist_theta(gen);

    // Add the noise to the predicted values
    current_particle.x = new_x + noise_x;
    current_particle.y = new_y + noise_y;
    current_particle.theta = new_theta + noise_theta;
  }); // end of for_each loop

} // end of prediction function

void ParticleFilter::dataAssociation(const std::vector<Map::single_landmark_s> predicted,
                                     std::vector<LandmarkObs> &observations, Particle &particle)
{
  /**
   * NOTE: Finds the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */

  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;

  std::for_each(std::begin(observations), std::end(observations), [&](LandmarkObs &current_observation) {
    // Calculate the distance between each observed measurement and all the predicted measurements
    vector<double> euclidean_distance;

    // Find the Euclidean distance between current predicted measurement and observed measurements
    // The result is stored in the euclidean_distance vector
    std::transform(predicted.begin(), predicted.end(), std::back_inserter(euclidean_distance),
                   [current_observation](Map::single_landmark_s const &current_predicted) {
                     return dist(current_observation.x, current_observation.y, current_predicted.x_f, current_predicted.y_f);
                   });

    // The landmark to which the observed measurement is closest is identified.
    // Its index in the landmark_within_range_list vector is stored as id in the observed measurement. This simplies the retrival process in updateWeights() function.
    int min_distance_landmark_index = std::min_element(euclidean_distance.begin(), euclidean_distance.end()) - euclidean_distance.begin(); // The index of the minimum distance element is found
    // The landmark vector index is assigned to the observed measurement index
    current_observation.id = min_distance_landmark_index;

    //For visualisation
    associations.push_back(predicted[min_distance_landmark_index].id_i);
    sense_x.push_back(current_observation.x);
    sense_y.push_back(current_observation.y);
  });

  SetAssociations(particle, associations, sense_x, sense_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * NOTE: Updates the weights of each particle using a mult-variate Gaussian 
   *   distribution. More information about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   The particles are located according to the MAP'S coordinate system. 
   *   There is a need to transform between the two systems. 
   *   This transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  std::for_each(particles.begin(), particles.end(), [&](Particle &current_particle) {
    // Performs the landmark selection within sensor range, transformation, association and weight calculation for each particle

    vector<LandmarkObs> observations_transformed;
    vector<Map::single_landmark_s> landmark_within_range_list;

    // Step 1: Select the landmarks that are within sensor range around each particle from the given landmarks (map_landmarks)
    std::copy_if(map_landmarks.landmark_list.begin(), map_landmarks.landmark_list.end(), std::back_inserter(landmark_within_range_list),
                 [sensor_range, current_particle](const Map::single_landmark_s current_landmark) {
                   return dist(current_particle.x, current_particle.y, current_landmark.x_f, current_landmark.y_f) <= sensor_range;
                 }); // end of step 1

    // Step 2: Transform the observed measurements from vehicle coordinates to map coordinates
    std::transform(observations.begin(), observations.end(), std::back_inserter(observations_transformed),
                   [current_particle](const LandmarkObs current_observation) {
                     // transform to map x coordinate
                     double x_map;
                     x_map = current_particle.x + (cos(current_particle.theta) * current_observation.x) - (sin(current_particle.theta) * current_observation.y);

                     // transform to map y coordinate
                     double y_map;
                     y_map = current_particle.y + (sin(current_particle.theta) * current_observation.x) + (cos(current_particle.theta) * current_observation.y);

                     LandmarkObs current_observation_transformed = {current_observation.id, x_map, y_map};

                     return current_observation_transformed;
                   }); // end of step 1

    // Step 3: Associate each observed measurement with corresponding landmark measurement
    this->dataAssociation(landmark_within_range_list, observations_transformed, current_particle); // end of step 3

    // Step 4: Compute the weight of each particle
    double final_weight = 1;
    std::for_each(observations_transformed.begin(), observations_transformed.end(), [this, &final_weight, &landmark_within_range_list, std_landmark, &current_particle](const LandmarkObs current_observation_transformed) {
      auto associated_landmark = landmark_within_range_list[current_observation_transformed.id];
      double weight = multivariate_gaussian_probability_density(std_landmark[0], std_landmark[1], current_observation_transformed.x, current_observation_transformed.y, associated_landmark.x_f, associated_landmark.y_f);
      if (weight > 0)
      {
        final_weight *= weight;
      }
    }); //end of step 4

    // Step 5: Update the weight of the particle
    current_particle.weight = final_weight;
    weights.push_back(final_weight); // Update the global weights vector
    // end of step 5
  }); //end of weight update for all particles
}

void ParticleFilter::resample()
{
  /**
   * NOTE: Resamples particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: std::discrete_distribution is helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::vector<Particle> new_particles; // Vector consisting of new particles

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i)
  {
    int resampled_particle_index = distribution(generator);
    new_particles.push_back(particles[resampled_particle_index]);
  }

  particles = new_particles;
  weights.clear();
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}