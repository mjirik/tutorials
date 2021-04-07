
# How the `PYTHONPATH` works

The modules imported without `.` are searched in directories in `PYTHONPATH`. It can be set by OS, but it is not necessary. 
It can be seen in `sys.path`

A current directory is added into `PYTHONPATH`. See few examples:
```shell
cd pymodules
```
* `python run.py` 
  
  add the actual (`pymodules`) dir
* `python mymodule/run_from_inside.py` 
  
  add `pymodules/mymodule` dir so it is too deep to see `mymodule` --> Failure

* `python -m mymodule.run_from_inside.py`

  add `pymodules` dir so it can see `mymodule`.

# ModuleNotFoundError: No module named 'mymodule'

1) Check `PYTHONPATH` by adding this in the beginning of your file. 

```python
import sys
print(sys.path)
```



I would not recommend to set `PYTHONPATH` in your os. Because it may cause problems with `conda`. However,
the [tutorial is available](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html)