# comhom
Python 3 library for compute homology group of cubical sets

# comhom3
Algorithms from Chapter 3 of Kaczynski, Mischaikow, and Mrozek (2003) "Computational Homology", Springer

# example of defining cubical set variables

interval(tuple)
 = (0,1)

cube(tuple of interval)
 = (
    (0,1),(3,4)
   )

cubicalSet(list of cube)
 = [
    ((0,1),(2,3)),
    ((0,0),(2,3))
   ]

chain(dictionary of cube)
 = {
    ((0,1),(3,4)) : 3,
    ((0,0),(3,4)) : 5,
    ((0,1),(3,3)) : -1,
    ((0,1),(4,4)) : 1,
   }
   
# NOTE
Index of array which represents matrix starts from 1 (1,2,3,...,N).

e.g. If you want to exchange 1st and 2nd row of matrix B, then rowExchange(B, 1, 2).

Lists, tuples, dictionaries etc. are indexed by non-negative sequence (0,1,2,...,N-1).
