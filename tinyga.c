/***************************************************************************
 *		tinyga.c - TinyGA
 *
 *	Mon, 01 Jun 2006 17:09:40 +0200
 *	Copyright 2006 Christophe Philemotte
 *	christophe.philemotte@8thcolor.com
 ***************************************************************************/
/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, US.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

//-----------------------------------------------------------------------------
//Parameters of the GA
#define POPSIZE 1000		//Size of the population, could be odd or even
#define LEN 128			//Length of the binary string which is used to represent the chromosom of an individual
#define GENERATIONS 10000	//The maximum number of generations
#define CROSSOVER_PROB 70	//The probability of crossing over of two selected parents in percentage
#define PMUT 10			//The probability of mutation in percentage per bit
#define GENERATIONAL 1		//1 for a generational system, 0 for a steady-state one

//Constants for using all the bits in a given type (portable, not linked to the endianess)
#define BLOCK unsigned int	//A given type is considered as a block of bits, you can use only unsigned type of char, short, int, long and long long
#define BLOCK_BIT (CHAR_BIT * sizeof (BLOCK))	//The number of bits in a block
#define NBR_OF_BLOCK ((LEN / BLOCK_BIT) + (LEN % BLOCK_BIT ? 1 : 0))	//The number of blocks that you need to storage a chromosom of lenght LEN
#define LOST_BITS ((NBR_OF_BLOCK * BLOCK_BIT) - LEN)	//The number of lost bits in the last block

//The main data
typedef BLOCK CHROM[NBR_OF_BLOCK];	//CHROM is an array of block and represent a chromosom.
typedef struct ga		//This structure contains all the main data we need to process the ga
{
  unsigned int *fit;		//An array of length LEN to store the fitness of all individuals
  unsigned int *cumfit;		//An array of length LEN to store the cumulative fitness of all individuals
  unsigned int sumfit;		//The sum of all fitnesses
  unsigned int bestIdx;		//The index of the best individual
  unsigned int worstIdx;	//The index of the worst individual
  CHROM *pop;			//An array of length POPSIZE to store the population of individuals (their chromosome)
  CHROM *nwpop;			//An temporary array of length POPSIZE (if generational) or 1 (if steady-state) to store the offspring before their insertion in the population
  unsigned int ngen;		//The number of generations
} GA;
static GA myGA;			//The data itself

//-----------------------------------------------------------------------------
//random functions
static inline unsigned int randInt (unsigned int max);	//Generate an unsigned integer between 0 and max following a uniform distribution
static inline unsigned int flip (unsigned int p);	//Flip a coin to obtain 1 with a probability p
unsigned int rouletteWheel (unsigned int seed);	//Generate an index between 0 and POPSIZE following a proportional fitness rule and a given fitness

//bit functions
static inline BLOCK readbits (BLOCK x, BLOCK p, BLOCK n);	//Read a number n of bits since the position p (the bits string x is indexed from 0 to BLOCK_BIT-1)
static inline BLOCK readbit (BLOCK x, BLOCK p);	//Read one bit at the position p in the bits string x
static inline BLOCK maskbit (BLOCK p);	//Generate a 0x0 to mask all bits of a string x except the one at p
static inline BLOCK flipbit (BLOCK x, BLOCK p);	//Flip a bit in the bits string x at p
unsigned int countbit (BLOCK x, unsigned int is_lb);	//Count the 1's bit in x and stop at BLOCK_BIT-LOST_BITS if is_lb is not null

//block functions
static inline BLOCK initBlock (unsigned int lb);	//Initialise a block x of bits from 0 to BLOCK_BIT-lb (uniform distribution)
BLOCK mutateBlock (BLOCK x, unsigned int lb);	//Mutate a block x of bits from 0 to BLOCK_BIT-lb with a probability per bit of PMUT

//chromosome functions
void initChrom (BLOCK * chro);	//Initialise a chromosome chro
inline static unsigned int getGene (BLOCK * chro, unsigned int idx);	//Return the gene value of a chromosome chro at idx
unsigned int fitness (BLOCK * chro);	//Compute the fitnese of a given chromosome chro

//genetic operators
void mutate (BLOCK * from);	//Mutate the chromosom chro and return the mutation
void crossover (BLOCK * p1, BLOCK * p2, BLOCK * c1, BLOCK * c2);	//Cross over two parents and generate two or one offspring (in this latter case, c2 has to be NULL)
void clone (BLOCK * from, BLOCK * to);	//Clone a chromosome to another one

//main functions of a ga
void initialise (void);		//Initialise randomly a population of chromosome uniformely distributed
void evaluate (void);		//Evaluate each individual
void selectAndReproduct (void);	//Generate the next generation: select the parents and produce the offsprings

//print functions
void printChrom (BLOCK * chro);	//Print a chromosome
void printParameters (void);	//Print the parameters of the ga
void printStatistics (void);	//Print the statistics of the ga

//memory functions
static inline void allocMem (void);	//Allocate memory for the data
static inline void freeMem (void);	//Free the used memory

//-----------------------------------------------------------------------------
//The Genetic Algorithm
int
main (int argc, char *argv[])
{
  if (argc > 1)
    srand (atoi (argv[1]));
  else
    srand (time (NULL));
  printParameters ();
  allocMem ();
  initialise ();
  evaluate ();
  printStatistics ();
  while (myGA.ngen < GENERATIONS && myGA.fit[myGA.bestIdx] < LEN)
    {
      unsigned int i;
      for (i = 0; i < (GENERATIONAL ? 1 : POPSIZE); ++i)
	{
	  selectAndReproduct ();
	  evaluate ();
	}
      ++myGA.ngen;
      printStatistics ();
    }
  if (myGA.fit[myGA.bestIdx] == LEN)
    printf ("SUCCESS\n");
  else
    printf ("FAILURE\n");
  freeMem ();
  return 0;
}

//-----------------------------------------------------------------------------
//random functions
static inline unsigned int
randInt (unsigned int max)
{
  return (rand () * ((float) max / RAND_MAX));
}

static inline unsigned int
flip (unsigned int p)
{
  if (p == 100)
    return 1;
  else if (p)
    return (randInt (100) <= p);
  else
    return 0;
}

unsigned int
rouletteWheel (unsigned int seed)
{
  unsigned int j, notfind = 1;
  for (j = 0; j < POPSIZE && notfind; j++)
    {
      if (myGA.cumfit[j] >= seed)
	notfind = 0;
    }
  return j - 1;
}

//-----------------------------------------------------------------------------
//bit functions
static inline BLOCK
readbits (BLOCK x, BLOCK p, BLOCK n)
{
  return (x >> (p + 1 - n)) & ~(((BLOCK) ~ 0x0UL) << n);
}

static inline BLOCK
readbit (BLOCK x, BLOCK p)
{
  return readbits (x, p, 1);
}

static inline BLOCK
maskbit (BLOCK p)
{
  return ~(((BLOCK) 0x1UL) << p);
}

static inline BLOCK
flipbit (BLOCK x, BLOCK p)
{
  return (x & maskbit (p)) | (~x & ~maskbit (p));
}

unsigned int
countbit (BLOCK x, unsigned int is_lb)
{
  unsigned int c;		// c accumulates the total bits set in v
  if (is_lb)
    x = readbits (x, BLOCK_BIT - LOST_BITS - 1, BLOCK_BIT - LOST_BITS);
  for (c = 0; x; c++)
    x &= x - 1;			// clear the least significant bit set
  return c;
}

//-----------------------------------------------------------------------------
//block functions
static inline BLOCK
initBlock (unsigned int lb)
{
  unsigned int i;
  BLOCK b = 0;
  for (i = 0; i < BLOCK_BIT - lb; ++i)
    {
      if (flip (50))
	b = flipbit (b, i);
    }
  return b;
}

BLOCK
mutateBlock (BLOCK x, unsigned int lb)
{
  unsigned int i;
  for (i = 0; i < BLOCK_BIT - lb; ++i)
    {
      if (flip (PMUT))
	x = flipbit (x, i);
    }
  return x;
}

//-----------------------------------------------------------------------------
//chromosome functions
void
initChrom (BLOCK * chro)
{
  unsigned int i;
  for (i = 0; i < NBR_OF_BLOCK - 1; ++i)
    {
      chro[i] = initBlock (0);
    }
  chro[i] = initBlock (LOST_BITS);
}

inline static unsigned int
getGene (BLOCK * chro, unsigned int idx)
{
  return readbit (chro[idx / BLOCK_BIT], idx % BLOCK_BIT);
}

unsigned int
fitness (BLOCK * chro)		//With the getGene function, you can define your function. Here, we have chosen a more efficient way to count the bit. But it is possible to use getGene to count the bits
{
  unsigned int i;
  unsigned int res = 0;
  for (i = 0; i < NBR_OF_BLOCK - 1; ++i)
    {
      res += countbit (chro[i], 0);
    }
  res += countbit (chro[i], 1);
  return res;
}

//-----------------------------------------------------------------------------
//genetic operators
void
mutate (BLOCK * from)
{
  unsigned int i;
  for (i = 0; i < NBR_OF_BLOCK - 1; ++i)
    {
      from[i] = mutateBlock (from[i], 0);
    }
  from[i] = mutateBlock (from[i], LOST_BITS);
}

void
crossover (BLOCK * p1, BLOCK * p2, BLOCK * c1, BLOCK * c2)
{
  static CHROM m;
  unsigned int i;
  initChrom (m);
  for (i = 0; i < NBR_OF_BLOCK; ++i)
    {
      c1[i] = (p1[i] & m[i]) | (p2[i] & ~m[i]);
      if (c2)
	c2[i] = (p1[i] & ~m[i]) | (p2[i] & m[i]);
    }
}

void
clone (BLOCK * from, BLOCK * to)
{
  unsigned int i;
  for (i = 0; i < NBR_OF_BLOCK; ++i)
    {
      to[i] = from[i];
    }
}

//-----------------------------------------------------------------------------
//main functions of a ga
void
initialise (void)
{
  unsigned int i;
  for (i = 0; i < POPSIZE; ++i)
    {
      initChrom (myGA.pop[i]);
    }
}

void
evaluate (void)
{
  unsigned int i;
  myGA.sumfit = 0;
  myGA.worstIdx = 0;
  myGA.bestIdx = 0;
  for (i = 0; i < POPSIZE; ++i)
    {
      myGA.fit[i] = fitness (myGA.pop[i]);
      myGA.sumfit += myGA.fit[i];
      myGA.cumfit[i] = myGA.sumfit;
      if (myGA.fit[i] > myGA.fit[myGA.bestIdx])
	myGA.bestIdx = i;
      else if (myGA.fit[i] < myGA.fit[myGA.worstIdx])
	myGA.worstIdx = i;
    }
}

void
selectAndReproduct (void)	//We have to tackle the generational and steady-state system. That's why we have a lot of ternary operators (GENERATIONAL ? : )
{
  unsigned int i, j, k;
  for (i = 0; i < (GENERATIONAL ? (POPSIZE / 2) : 1); ++i)
    {
      j = rouletteWheel (randInt (myGA.sumfit));
      k = rouletteWheel (randInt (myGA.sumfit));
      if (flip (CROSSOVER_PROB))
	crossover (myGA.pop[j], myGA.pop[k], myGA.nwpop[i * 2],
		   (GENERATIONAL ? myGA.nwpop[i * 2 + 1] : NULL));
      else
	{
	  clone (myGA.pop[j], myGA.nwpop[i * 2]);
	  if (GENERATIONAL)
	    clone (myGA.pop[k], myGA.nwpop[i * 2 + 1]);
	}
    }
  if (GENERATIONAL && POPSIZE % 2)
    {
      j = rouletteWheel (randInt (myGA.sumfit));
      clone (myGA.pop[j], myGA.nwpop[POPSIZE - 1]);
    }
  for (i = 0; i < (GENERATIONAL ? POPSIZE : 1); ++i)
    {
      mutate (myGA.nwpop[i]);
      clone (myGA.nwpop[i], myGA.pop[(GENERATIONAL ? i : myGA.worstIdx)]);
    }
}

//-----------------------------------------------------------------------------
//print functions
void
printChrom (BLOCK * chro)
{
  unsigned int i;
  for (i = 0; i < LEN; ++i)
    putchar ('0' + getGene (chro, i));
}

void
printParameters (void)
{
  printf("LEN\t%d\nPOPSIZE\t%d\nGENERATIONS\t%d\nCROSSOVER_PROB\t%d\nPMUT\t%d\nGENERATIONAL\t%s\nGeneration Number\tAverage Fitness\tBest Fitness\tBest Individual\t", LEN, POPSIZE, GENERATIONS, CROSSOVER_PROB, PMUT, (GENERATIONAL ? "true" : "false"));
}

void
printStatistics (void)
{
  printf("%d\t%f\t%d\t", myGA.ngen, (float)myGA.sumfit / POPSIZE, myGA.fit[myGA.bestIdx]);
  printChrom (myGA.pop[myGA.bestIdx]);
  putchar('\n');
}

//-----------------------------------------------------------------------------
//memory functions
static inline void
allocMem (void)
{
  myGA.fit = calloc (POPSIZE, sizeof (unsigned int));
  myGA.cumfit = calloc (POPSIZE, sizeof (unsigned int));
  myGA.pop = calloc (POPSIZE, sizeof (CHROM));
  myGA.nwpop = calloc ((GENERATIONAL ? POPSIZE : 1), sizeof (CHROM));
  myGA.ngen = 0, myGA.bestIdx = 0, myGA.worstIdx = 0;
}

static inline void
freeMem (void)
{
  free (myGA.pop);
  free (myGA.nwpop);
  free (myGA.fit);
  free (myGA.cumfit);
}
