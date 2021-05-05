import threading
import time


def mythread():
    time.sleep(1000)

def main():
    threads = 0     #thread counter
    y = 1000000     #a MILLION of 'em!
    for i in range(y):
        try:
            x = threading.Thread(target=mythread, daemon=True)
            threads += 1    #thread counter
            x.start()       #start each thread
        except RuntimeError:    #too many throws RuntimeError
            break
    print("{} threads created.\n".format(threads))

if __name__ == "__main__":
    main()




# import math, sys, time
# import pp
#
# def isprime(n):
#     print(time.time())
#     if n == 2 or n == 3: return True
#     if n < 2 or n % 2 == 0: return False
#     if n < 9: return True
#     if n % 3 == 0: return False
#     r = int(n ** 0.5)
#     # since all primes > 3 are of the form 6n ± 1
#     # start with f=5 (which is prime)
#     # and test f, f+2 for being prime
#     # then loop by 6.
#     f = 5
#     while f <= r:
#         # print('\t', f)
#         if n % f == 0: return False
#         if n % (f + 2) == 0: return False
#         f += 6
#     return True
#
# def sum_primes(n):
#     """Calculates sum of all primes below given integer n"""
#     return sum([x for x in range(2,n) if isprime(x)])
#
#
# # tuple of all parallel python servers to connect with
# ppservers = ()
# #ppservers = ("10.0.0.1",)
#
# if len(sys.argv) > 1:
#     ncpus = int(sys.argv[1])
#     # Creates jobserver with ncpus workers
#     job_server = pp.Server(ncpus, ppservers=ppservers)
# else:
#     # Creates jobserver with automatically detected number of workers
#     job_server = pp.Server(ppservers=ppservers)
#
# print("Starting pp with", job_server.get_ncpus(), "workers")
#
# # Submit a job of calulating sum_primes(100) for execution.
# # sum_primes - the function
# # (100,) - tuple with arguments for sum_primes
# # (isprime,) - tuple with functions on which function sum_primes depends
# # ("math",) - tuple with module names which must be imported before sum_primes execution
# # Execution starts as soon as one of the workers will become available
# job1 = job_server.submit(sum_primes, (100,), (isprime,), ("time",))
#
# # Retrieves the result calculated by job1
# # The value of job1() is the same as sum_primes(100)
# # If the job has not been finished yet, execution will wait here until result is available
# result = job1()
#
# print("Sum of primes below 100 is", result)
#
# start_time = time.time()
#
# # The following submits 8 jobs and then retrieves the results
# inputs = (1000000, 1000100, 1000200, 1000300, 1000400, 1000500, 1000600, 1000700)
# jobs = [(input, job_server.submit(sum_primes,(input,), (isprime,), ("math",))) for input in inputs]
# for input, job in jobs:
#     print("Sum of primes below", input, "is", job())
#
# print("Time elapsed: ", time.time() - start_time, "s")
# job_server.print_stats()
