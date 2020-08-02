/* author: Cornelius Marx
 * description: Small library providing useful math functions
 */

#ifndef F_MATH_H
#define F_MATH_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

//often used math-functions
#ifdef __cplusplus
extern "C" {
#endif

/* returning maximum/minimum of two given values
 * @return a if a>b else b
 */
int f_imax(int a, int b);
int f_imin(int a, int b);
float f_fmax(float a, float b);
float f_fmin(float a, float b);

/* limiting given value to upper and lower bound
 * @return max if a > max, min if a < min, else a
 */
int f_ilimit(int a, int max, int min);
float f_flimit(float a, float max, float min);

/* round to next integer
 * @return rounded value of a
 */
float f_round(float a);

/* signum function
 * @return 1 if a is positive, -1 if a is negative or 0 if a is 0
 */
float f_fsign(float a);
int f_isign(int a);

/* converting radiants to degrees and vice versa
 */
float f_rad(float deg);
float f_deg(float rad);

/* check if two values a and b are equal to a given amount
 * @param a first value
 * @param b second value
 * @param epsilon 
 * @return 1 if fabs(a-b)<=epsilon else 0
 */
int f_equals(float a, float b, float epsilon);

/* perform a quadratic remapping of linear intervall [0, 1]
 * @param t value in [0, 1]
 * @return quadratic interpolated value of t
 */
float f_quadricInterpolate(float t);

/* random value between 0 and 1
 * @return random value in [0, 1]
 * NOTE: this function uses the standard math function rand(), the seed can be set with srand()
 */
double f_random();

/* returns a random value in [min, max]
 * @param min minimum value to return
 * @param max maximum value to return
 * @return random value in [min, max]
 * NOTE: this function uses the standard math function rand(), the seed can be set with srand()
 */
int f_irandomRange(int min, int max);
float f_frandomRange(float min, float max);

/* select a random index according to frequencies in a given array
 * @param bucket_freq array of floats, each value corresponds to a frequency of its index relative to the total sum of frequencies
 * @param num_buckets size of array bucket_freq
 * @param sum if sum != NULL this value is used as total sum of all frequencies, otherwise it the total sum is calculated in this function
 * @return random index in [0, num_buckets-1]
 * @example bucket_freq=[5, 1.5, 3.5], index 0 will is with a chance of 50%, index 1 is with a chance of 15%, index 2 is returned with a chance of 35%
 * NOTE: this function uses the standard math function rand(), the seed can be set with srand()
 */
int f_randomBuckets(const float * bucket_freq, int num_buckets, const float *sum);

/* selection sort values of a given array in ascending order (array[0] smallest value, array[size-1] largest value)
 * @param array the array to sort
 * @param size the size of array
 */
void f_selectionSort(float * array, int size);

#ifdef __cplusplus
}
#endif
#endif
