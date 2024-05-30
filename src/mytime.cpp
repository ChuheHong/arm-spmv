/**
 * @file    : mytime.h
 * @author  : theSparky Team
 * @version :
 * 
 * Functions for time.
 */
#include "mytime.h" 
	
#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>
double mytimer(void) 
{
	struct timeval tp;
	static long start=0, startu;
	if (!start) {
		gettimeofday(&tp, NULL);
		start = tp.tv_sec;
		startu = tp.tv_usec;
		return 0.0;
	}
	gettimeofday(&tp, NULL);
	return ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 ;
}


