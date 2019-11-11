---
layout: post
title: Joblib and Numba
date: 2019-08-01 00:00:00
description: Guide to easy speed increases in Python that can be implemented using only simple decorators to user-defined functions.
img: paralell.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Python, Performance, Coding]
---

This post is based on a presentation given at King's College London for a session aimed at PhD students who use computational tools in their research. The aim was to agree on a good development strategy in terms of performance, version control and testing. You can find the <a href="{{site.baseurl}}/assets/pdf/free-speed.pdf" target="_blank"><i class="fa fa-file-pdf-o"></i> PDF</a> here.

## Numba Python Library

The main reference for this page is just the [Numba Website](https://numba.pydata.org/) which has links to all the reference material. We present just a set of minimal examples that can be implemented to find easy speed increases in Python code. Most of what follows is understanding how to use some very simple decorators on your already written functions. These include;

* `@jit` "Just In Time" compilation decorator which optimizes the calculations especially if type information is given.
* `@vectorize` Many functions in numpy are “universal” in the sense that they can apply element-wise operations to arrays, this is a way of generating your own.

### `nopython` Mode

The Numba `@jit` decorator fundamentally operates in two compilation modes, `nopython` mode and `object` mode. The behaviour of the `nopython` compilation mode is to essentially compile the decorated function so that it will run entirely without the involvement of the Python interpreter. This is the recommended and best-practice way to use the Numba jit decorator as it leads to the best performance.

{% highlight python %}
from numba import jit
import numpy as np
import time

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
{% endhighlight %}
{% highlight python %}
>> Elapsed (with compilation) = 0.1720
>> Elapsed (after compilation) = 4.220e-05
{% endhighlight %}

### How does Numba Work?

Numba reads the Python bytecode for a decorated function and combines this with information about the types of the input arguments to the function. It analyzes and optimizes your code, and finally uses the LLVM compiler library to generate a machine code version of your function, tailored to your CPU capabilities. There is also support for Nvidia CUDA GPUs. This compiled version is then used every time your function is called.

### Other Decorators

* `@njit` - this is an alias for `@jit(nopython=True)` as it is so commonly used!
* `@vectorize` - produces NumPy ufunc s (with all the ufunc methods supported).
* `@guvectorize` - produces NumPy generalized ufunc s. Docs are here.
* `@stencil` - declare a function as a kernel for a stencil like operation.
* `@jitclass` - for jit aware classes.
* `@cfunc` - declare a function for use as a native call back (to be called from C/C++ etc).
* `@overload` - register your own implementation of a function for use in `nopython` mode, e.g. `@overload(scipy.special.j0)`.

### Extra Options

* `paralell = True` - enables the automatic parallelization of the function.
* `fastmath = True` - enables fast-math behaviour for the function.

### Compiling Python code with `@jit`

The recommended way to use the `@jit` decorator is to let Numba decide when and how to optimize. For example,

{% highlight python %}
from numba import jit

@jit
def f(x, y):
    return x + y
{% endhighlight %}

You can also tell Numba the function signature you are expecting. The function `f(x, y)` would now look like:

{% highlight python %}
from numba import jit, int32

@jit(int32(int32, int32))
def f(x, y):
    return x + y

# The higher order bits get discarded so can lead
# to counterintuitive results
f(2**31, 2**31 + 1)
>> 1
{% endhighlight %}

`int32(int32, int32)` is the function’s signature. In this case, the corresponding specialization will be compiled by the `@jit` decorator, and no other specialization will be allowed. This is useful if you want fine-grained control over types chosen by the compiler (for example, to use single-precision floats).

### Compilation Options

1. `nopython` Numba has two compilation modes: nopython mode and object mode. The former produces much faster code, but has limitations that can force Numba to fall back to the latter. To prevent Numba from falling back, and instead raise an error, pass `nopython=True`.
2. `cache` To avoid compilation times each time you invoke a Python program, you can instruct Numba to write the result of function compilation into a file-based cache. This is done by passing `cache=True`.
3. `parallel` Enables automatic parallelization (and related optimizations) for those operations in the function known to have parallel semantics. This feature is enabled by passing `parallel=True` and must be used in conjunction with `nopython=True`.

### Creating NumPy Universal Functions

Numba’s vectorize allows Python functions taking scalar input arguments to be used as NumPy ufuncs. Creating a traditional NumPy ufunc is not the most straightforward process and involves writing some C code. Numba makes this easy. Using the `vectorize()` decorator, Numba can compile a pure Python function into a ufunc that operates over NumPy arrays as fast as traditional ufuncs written in C.

Using `vectorize()`, you write your function as operating over input scalars, rather than arrays. Numba will generate the surrounding loop (or kernel) allowing efficient iteration over the actual inputs.

The `vectorize()` decorator has two modes of operation:

* Eager, or decoration-time, compilation: If you pass one or more type signatures to the decorator, you will be building a Numpy universal function (ufunc). The rest of this subsection describes building ufuncs using decoration-time compilation.
* Lazy, or call-time, compilation: When not given any signatures, the decorator will give you a Numba dynamic universal function (DUFunc) that dynamically compiles a new kernel when called with a previously unsupported input type.

As described above, if you pass a list of signatures to the vectorize() decorator, your function will be compiled into a Numpy ufunc. In the basic case, only one signature will be passed:

{% highlight python %}
from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def f(x, y):
    return x + y
{% endhighlight %}

If you pass several signatures, beware that you have to pass most specific signatures before least specific ones (e.g., single-precision floats before double-precision floats), otherwise type-based dispatching will not work as expected:

{% highlight python %}
from numba import vectorize, int32, int64, float32, float64
@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def f(x, y):
    return x + y
{% endhighlight %}

To see a simple example consider the following code:

{% highlight python %}
def not_vectorized(x):
    r"""
    Here, x is an array that we will perform an element wise operation on.
    This is the case where the operation is not vectorized.

    Parameters
    ----------
    x : float
            input

    Returns
    -------
    y : float
            output
    """
    if x > 0:
            y = 1
    else:
            y = 0
    return y

@vectorize(nopython=True)
def vectorized(x):
    r"""
    Here, x is an array that we will perform an element wise operation on. 
    This is the case where the operation is vectorized.

    Parameters
    ----------
    x : float or array
            input

    Returns
    -------
    y : float or array
            output
    """
    if x > 0:
            y = 1
    else:
            y = 0
    return y
{% endhighlight %}

{% highlight python %}
>> x = 1 gives y = 1

>> Now we try with x = [-1  0  1]
>> Calling not_vectorized(x) gives the following error
>> ERROR: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
>> We see that the operation does not work
>> Now we try the same with the vectorized version.
>> As before with x = 1, a float, vectorized(x) gives y = 1
>> Now we try with x = [-1  0  1]
>> Calling vectorized(x) gives the correct result!
>> vectorized(x) = [0 0 1]
{% endhighlight %}

## The Joblib Library

Finally, we provide a very simple example of how to parallelise for loops in Python. The exact improvement you get of course depends on the number of cores you have, but running on a cluster or a good desktop, you can find improvements of 10-100x with very little effort. To illustrate this we show how to transform a for loop into a parallelised form.

{% highlight python %}
from joblib import Parallel, delayed

def add(x, y):
    return x + y

# For loop:
result = []
for x, y in zip([0, 1, 2], [4, 5, 6]):
    result.append(add(x, y))
    
# Parallelised:
result = Parallel(n_jobs=-1)(delayed(add)(x, y) for (x, y) in zip([0, 1, 2], [4, 5, 6]))
{% endhighlight %}

<a href="{{site.baseurl}}/"><i class="fa fa-home" aria-hidden="true"></i> Homepage</a>