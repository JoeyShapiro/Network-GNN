??
? ? 
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
?
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
?
graph_convolution/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namegraph_convolution/kernel
?
,graph_convolution/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution/kernel*
_output_shapes

: *
dtype0
?
graph_convolution_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_namegraph_convolution_1/kernel
?
.graph_convolution_1/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution_1/kernel*
_output_shapes

:  *
dtype0
?
graph_convolution_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_namegraph_convolution_2/kernel
?
.graph_convolution_2/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution_2/kernel*
_output_shapes

:  *
dtype0
?
graph_convolution_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_namegraph_convolution_3/kernel
?
.graph_convolution_3/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution_3/kernel*
_output_shapes

: *
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:a*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
: *
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
: *
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/graph_convolution/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/graph_convolution/kernel/m
?
3Adam/graph_convolution/kernel/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/kernel/m*
_output_shapes

: *
dtype0
?
!Adam/graph_convolution_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!Adam/graph_convolution_1/kernel/m
?
5Adam/graph_convolution_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_1/kernel/m*
_output_shapes

:  *
dtype0
?
!Adam/graph_convolution_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!Adam/graph_convolution_2/kernel/m
?
5Adam/graph_convolution_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_2/kernel/m*
_output_shapes

:  *
dtype0
?
!Adam/graph_convolution_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!Adam/graph_convolution_3/kernel/m
?
5Adam/graph_convolution_3/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_3/kernel/m*
_output_shapes

: *
dtype0
?
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*%
shared_nameAdam/conv1d/kernel/m
?
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:a*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_1/kernel/m
?
*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
: *
dtype0
?
Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/graph_convolution/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/graph_convolution/kernel/v
?
3Adam/graph_convolution/kernel/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/kernel/v*
_output_shapes

: *
dtype0
?
!Adam/graph_convolution_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!Adam/graph_convolution_1/kernel/v
?
5Adam/graph_convolution_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_1/kernel/v*
_output_shapes

:  *
dtype0
?
!Adam/graph_convolution_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!Adam/graph_convolution_2/kernel/v
?
5Adam/graph_convolution_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_2/kernel/v*
_output_shapes

:  *
dtype0
?
!Adam/graph_convolution_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!Adam/graph_convolution_3/kernel/v
?
5Adam/graph_convolution_3/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_3/kernel/v*
_output_shapes

: *
dtype0
?
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*%
shared_nameAdam/conv1d/kernel/v
?
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:a*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_1/kernel/v
?
*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
: *
dtype0
?
Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Y
value?YB?Y B?Y
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
 
^

kernel
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
^

(kernel
)	variables
*regularization_losses
+trainable_variables
,	keras_api
R
-	variables
.regularization_losses
/trainable_variables
0	keras_api
^

1kernel
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
^

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api

?	keras_api
 
R
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
R
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
h

Nkernel
Obias
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
R
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
h

Xkernel
Ybias
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
h

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
?
hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratem?(m?1m?:m?Dm?Em?Nm?Om?Xm?Ym?bm?cm?v?(v?1v?:v?Dv?Ev?Nv?Ov?Xv?Yv?bv?cv?
V
0
(1
12
:3
D4
E5
N6
O7
X8
Y9
b10
c11
 
V
0
(1
12
:3
D4
E5
N6
O7
X8
Y9
b10
c11
?
	variables
regularization_losses
mnon_trainable_variables
trainable_variables
nlayer_regularization_losses

olayers
player_metrics
qmetrics
 
 
 
 
?
	variables
regularization_losses
rnon_trainable_variables
trainable_variables
slayer_regularization_losses

tlayers
ulayer_metrics
vmetrics
db
VARIABLE_VALUEgraph_convolution/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
 	variables
!regularization_losses
wnon_trainable_variables
"trainable_variables
xlayer_regularization_losses

ylayers
zlayer_metrics
{metrics
 
 
 
?
$	variables
%regularization_losses
|non_trainable_variables
&trainable_variables
}layer_regularization_losses

~layers
layer_metrics
?metrics
fd
VARIABLE_VALUEgraph_convolution_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

(0
 

(0
?
)	variables
*regularization_losses
?non_trainable_variables
+trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
 
 
 
?
-	variables
.regularization_losses
?non_trainable_variables
/trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
fd
VARIABLE_VALUEgraph_convolution_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

10
 

10
?
2	variables
3regularization_losses
?non_trainable_variables
4trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
 
 
 
?
6	variables
7regularization_losses
?non_trainable_variables
8trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
fd
VARIABLE_VALUEgraph_convolution_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE

:0
 

:0
?
;	variables
<regularization_losses
?non_trainable_variables
=trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
 
 
 
 
?
@	variables
Aregularization_losses
?non_trainable_variables
Btrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
?
F	variables
Gregularization_losses
?non_trainable_variables
Htrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
 
 
 
?
J	variables
Kregularization_losses
?non_trainable_variables
Ltrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
?
P	variables
Qregularization_losses
?non_trainable_variables
Rtrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
 
 
 
?
T	variables
Uregularization_losses
?non_trainable_variables
Vtrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1
 

X0
Y1
?
Z	variables
[regularization_losses
?non_trainable_variables
\trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
 
 
 
?
^	variables
_regularization_losses
?non_trainable_variables
`trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
 

b0
c1
?
d	variables
eregularization_losses
?non_trainable_variables
ftrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/graph_convolution/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/graph_convolution/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*4
_output_shapes"
 :??????????????????*
dtype0*)
shape :??????????????????
?
serving_default_input_2Placeholder*0
_output_shapes
:??????????????????*
dtype0
*%
shape:??????????????????
?
serving_default_input_3Placeholder*=
_output_shapes+
):'???????????????????????????*
dtype0*2
shape):'???????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3graph_convolution/kernelgraph_convolution_1/kernelgraph_convolution_2/kernelgraph_convolution_3/kernelconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_8235
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,graph_convolution/kernel/Read/ReadVariableOp.graph_convolution_1/kernel/Read/ReadVariableOp.graph_convolution_2/kernel/Read/ReadVariableOp.graph_convolution_3/kernel/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/graph_convolution/kernel/m/Read/ReadVariableOp5Adam/graph_convolution_1/kernel/m/Read/ReadVariableOp5Adam/graph_convolution_2/kernel/m/Read/ReadVariableOp5Adam/graph_convolution_3/kernel/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp3Adam/graph_convolution/kernel/v/Read/ReadVariableOp5Adam/graph_convolution_1/kernel/v/Read/ReadVariableOp5Adam/graph_convolution_2/kernel/v/Read/ReadVariableOp5Adam/graph_convolution_3/kernel/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_9661
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegraph_convolution/kernelgraph_convolution_1/kernelgraph_convolution_2/kernelgraph_convolution_3/kernelconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/graph_convolution/kernel/m!Adam/graph_convolution_1/kernel/m!Adam/graph_convolution_2/kernel/m!Adam/graph_convolution_3/kernel/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/graph_convolution/kernel/v!Adam/graph_convolution_1/kernel/v!Adam/graph_convolution_2/kernel/v!Adam/graph_convolution_3/kernel/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_9806??
?
?
$__inference_model_layer_call_fn_8950
inputs_0
inputs_1

inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_81672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?
{
&__inference_dense_1_layer_call_fn_9501

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_79702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7451

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :?????????????????? 2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_9471

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
cond_true_9332
cond_sub_shape3
/cond_pad_map_tensorarrayv2stack_tensorliststack
cond_identityZ

cond/sub/xConst*
_output_shapes
: *
dtype0*
value	B :#2

cond/sub/xe
cond/subSubcond/sub/x:output:0cond_sub_shape*
T0*
_output_shapes
:2

cond/sub~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack?
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1?
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2?
cond/strided_sliceStridedSlicecond/sub:z:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicep
cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Pad/paddings/1/0?
cond/Pad/paddings/1Packcond/Pad/paddings/1/0:output:0cond/strided_slice:output:0*
N*
T0*
_output_shapes
:2
cond/Pad/paddings/1
cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2
cond/Pad/paddings/0_1
cond/Pad/paddings/2_1Const*
_output_shapes
:*
dtype0*
valueB"        2
cond/Pad/paddings/2_1?
cond/Pad/paddingsPackcond/Pad/paddings/0_1:output:0cond/Pad/paddings/1:output:0cond/Pad/paddings/2_1:output:0*
N*
T0*
_output_shapes

:2
cond/Pad/paddings?
cond/PadPad/cond_pad_map_tensorarrayv2stack_tensorliststackcond/Pad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????a2

cond/Pad|
cond/IdentityIdentitycond/Pad:output:0*
T0*4
_output_shapes"
 :??????????????????a2
cond/Identity"'
cond_identitycond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?
U
+__inference_sort_pooling_layer_call_fn_9375

embeddings
mask

identity?
PartitionedCallPartitionedCall
embeddingsmask*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sort_pooling_layer_call_and_return_conditional_losses_78162
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????????????a:??????????????????:` \
4
_output_shapes"
 :??????????????????a
$
_user_specified_name
embeddings:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
D
(__inference_dropout_3_layer_call_fn_9166

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_75932
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_9151

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
Ӛ
?	
&model_sort_pooling_map_while_body_7153J
Fmodel_sort_pooling_map_while_model_sort_pooling_map_while_loop_counterE
Amodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice,
(model_sort_pooling_map_while_placeholder.
*model_sort_pooling_map_while_placeholder_1I
Emodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice_1_0?
?model_sort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0?
?model_sort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0)
%model_sort_pooling_map_while_identity+
'model_sort_pooling_map_while_identity_1+
'model_sort_pooling_map_while_identity_2+
'model_sort_pooling_map_while_identity_3G
Cmodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice_1?
model_sort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor?
?model_sort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor?
Nmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2P
Nmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shape?
@model/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?model_sort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0(model_sort_pooling_map_while_placeholderWmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????a*
element_dtype02B
@model/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem?
Pmodel/sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2R
Pmodel/sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
Bmodel/sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?model_sort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0(model_sort_pooling_map_while_placeholderYmodel/sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*#
_output_shapes
:?????????*
element_dtype0
2D
Bmodel/sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem?
/model/sort_pooling/map/while/boolean_mask/ShapeShapeGmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:21
/model/sort_pooling/map/while/boolean_mask/Shape?
=model/sort_pooling/map/while/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=model/sort_pooling/map/while/boolean_mask/strided_slice/stack?
?model/sort_pooling/map/while/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?model/sort_pooling/map/while/boolean_mask/strided_slice/stack_1?
?model/sort_pooling/map/while/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?model/sort_pooling/map/while/boolean_mask/strided_slice/stack_2?
7model/sort_pooling/map/while/boolean_mask/strided_sliceStridedSlice8model/sort_pooling/map/while/boolean_mask/Shape:output:0Fmodel/sort_pooling/map/while/boolean_mask/strided_slice/stack:output:0Hmodel/sort_pooling/map/while/boolean_mask/strided_slice/stack_1:output:0Hmodel/sort_pooling/map/while/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:29
7model/sort_pooling/map/while/boolean_mask/strided_slice?
@model/sort_pooling/map/while/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2B
@model/sort_pooling/map/while/boolean_mask/Prod/reduction_indices?
.model/sort_pooling/map/while/boolean_mask/ProdProd@model/sort_pooling/map/while/boolean_mask/strided_slice:output:0Imodel/sort_pooling/map/while/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 20
.model/sort_pooling/map/while/boolean_mask/Prod?
1model/sort_pooling/map/while/boolean_mask/Shape_1ShapeGmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1model/sort_pooling/map/while/boolean_mask/Shape_1?
?model/sort_pooling/map/while/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?model/sort_pooling/map/while/boolean_mask/strided_slice_1/stack?
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1?
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2?
9model/sort_pooling/map/while/boolean_mask/strided_slice_1StridedSlice:model/sort_pooling/map/while/boolean_mask/Shape_1:output:0Hmodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack:output:0Jmodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1:output:0Jmodel/sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2;
9model/sort_pooling/map/while/boolean_mask/strided_slice_1?
1model/sort_pooling/map/while/boolean_mask/Shape_2ShapeGmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1model/sort_pooling/map/while/boolean_mask/Shape_2?
?model/sort_pooling/map/while/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?model/sort_pooling/map/while/boolean_mask/strided_slice_2/stack?
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1?
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Amodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2?
9model/sort_pooling/map/while/boolean_mask/strided_slice_2StridedSlice:model/sort_pooling/map/while/boolean_mask/Shape_2:output:0Hmodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack:output:0Jmodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1:output:0Jmodel/sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2;
9model/sort_pooling/map/while/boolean_mask/strided_slice_2?
9model/sort_pooling/map/while/boolean_mask/concat/values_1Pack7model/sort_pooling/map/while/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2;
9model/sort_pooling/map/while/boolean_mask/concat/values_1?
5model/sort_pooling/map/while/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model/sort_pooling/map/while/boolean_mask/concat/axis?
0model/sort_pooling/map/while/boolean_mask/concatConcatV2Bmodel/sort_pooling/map/while/boolean_mask/strided_slice_1:output:0Bmodel/sort_pooling/map/while/boolean_mask/concat/values_1:output:0Bmodel/sort_pooling/map/while/boolean_mask/strided_slice_2:output:0>model/sort_pooling/map/while/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model/sort_pooling/map/while/boolean_mask/concat?
1model/sort_pooling/map/while/boolean_mask/ReshapeReshapeGmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:09model/sort_pooling/map/while/boolean_mask/concat:output:0*
T0*'
_output_shapes
:?????????a23
1model/sort_pooling/map/while/boolean_mask/Reshape?
9model/sort_pooling/map/while/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2;
9model/sort_pooling/map/while/boolean_mask/Reshape_1/shape?
3model/sort_pooling/map/while/boolean_mask/Reshape_1ReshapeImodel/sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem:item:0Bmodel/sort_pooling/map/while/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????25
3model/sort_pooling/map/while/boolean_mask/Reshape_1?
/model/sort_pooling/map/while/boolean_mask/WhereWhere<model/sort_pooling/map/while/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????21
/model/sort_pooling/map/while/boolean_mask/Where?
1model/sort_pooling/map/while/boolean_mask/SqueezeSqueeze7model/sort_pooling/map/while/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
23
1model/sort_pooling/map/while/boolean_mask/Squeeze?
7model/sort_pooling/map/while/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/sort_pooling/map/while/boolean_mask/GatherV2/axis?
2model/sort_pooling/map/while/boolean_mask/GatherV2GatherV2:model/sort_pooling/map/while/boolean_mask/Reshape:output:0:model/sort_pooling/map/while/boolean_mask/Squeeze:output:0@model/sort_pooling/map/while/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:?????????a24
2model/sort_pooling/map/while/boolean_mask/GatherV2?
0model/sort_pooling/map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????22
0model/sort_pooling/map/while/strided_slice/stack?
2model/sort_pooling/map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2model/sort_pooling/map/while/strided_slice/stack_1?
2model/sort_pooling/map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2model/sort_pooling/map/while/strided_slice/stack_2?
*model/sort_pooling/map/while/strided_sliceStridedSlice;model/sort_pooling/map/while/boolean_mask/GatherV2:output:09model/sort_pooling/map/while/strided_slice/stack:output:0;model/sort_pooling/map/while/strided_slice/stack_1:output:0;model/sort_pooling/map/while/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2,
*model/sort_pooling/map/while/strided_slice?
)model/sort_pooling/map/while/argsort/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model/sort_pooling/map/while/argsort/axis?
*model/sort_pooling/map/while/argsort/ShapeShape3model/sort_pooling/map/while/strided_slice:output:0*
T0*
_output_shapes
:2,
*model/sort_pooling/map/while/argsort/Shape?
8model/sort_pooling/map/while/argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model/sort_pooling/map/while/argsort/strided_slice/stack?
:model/sort_pooling/map/while/argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/sort_pooling/map/while/argsort/strided_slice/stack_1?
:model/sort_pooling/map/while/argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/sort_pooling/map/while/argsort/strided_slice/stack_2?
2model/sort_pooling/map/while/argsort/strided_sliceStridedSlice3model/sort_pooling/map/while/argsort/Shape:output:0Amodel/sort_pooling/map/while/argsort/strided_slice/stack:output:0Cmodel/sort_pooling/map/while/argsort/strided_slice/stack_1:output:0Cmodel/sort_pooling/map/while/argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model/sort_pooling/map/while/argsort/strided_slice?
)model/sort_pooling/map/while/argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2+
)model/sort_pooling/map/while/argsort/Rank?
+model/sort_pooling/map/while/argsort/TopKV2TopKV23model/sort_pooling/map/while/strided_slice:output:0;model/sort_pooling/map/while/argsort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????2-
+model/sort_pooling/map/while/argsort/TopKV2?
*model/sort_pooling/map/while/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model/sort_pooling/map/while/GatherV2/axis?
%model/sort_pooling/map/while/GatherV2GatherV2Gmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:05model/sort_pooling/map/while/argsort/TopKV2:indices:03model/sort_pooling/map/while/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????a2'
%model/sort_pooling/map/while/GatherV2?
"model/sort_pooling/map/while/ShapeShapeGmodel/sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"model/sort_pooling/map/while/Shape?
2model/sort_pooling/map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model/sort_pooling/map/while/strided_slice_1/stack?
4model/sort_pooling/map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model/sort_pooling/map/while/strided_slice_1/stack_1?
4model/sort_pooling/map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model/sort_pooling/map/while/strided_slice_1/stack_2?
,model/sort_pooling/map/while/strided_slice_1StridedSlice+model/sort_pooling/map/while/Shape:output:0;model/sort_pooling/map/while/strided_slice_1/stack:output:0=model/sort_pooling/map/while/strided_slice_1/stack_1:output:0=model/sort_pooling/map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model/sort_pooling/map/while/strided_slice_1?
$model/sort_pooling/map/while/Shape_1Shape.model/sort_pooling/map/while/GatherV2:output:0*
T0*
_output_shapes
:2&
$model/sort_pooling/map/while/Shape_1?
2model/sort_pooling/map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model/sort_pooling/map/while/strided_slice_2/stack?
4model/sort_pooling/map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model/sort_pooling/map/while/strided_slice_2/stack_1?
4model/sort_pooling/map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model/sort_pooling/map/while/strided_slice_2/stack_2?
,model/sort_pooling/map/while/strided_slice_2StridedSlice-model/sort_pooling/map/while/Shape_1:output:0;model/sort_pooling/map/while/strided_slice_2/stack:output:0=model/sort_pooling/map/while/strided_slice_2/stack_1:output:0=model/sort_pooling/map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model/sort_pooling/map/while/strided_slice_2?
 model/sort_pooling/map/while/subSub5model/sort_pooling/map/while/strided_slice_1:output:05model/sort_pooling/map/while/strided_slice_2:output:0*
T0*
_output_shapes
: 2"
 model/sort_pooling/map/while/sub?
-model/sort_pooling/map/while/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2/
-model/sort_pooling/map/while/Pad/paddings/0/0?
+model/sort_pooling/map/while/Pad/paddings/0Pack6model/sort_pooling/map/while/Pad/paddings/0/0:output:0$model/sort_pooling/map/while/sub:z:0*
N*
T0*
_output_shapes
:2-
+model/sort_pooling/map/while/Pad/paddings/0?
-model/sort_pooling/map/while/Pad/paddings/1_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-model/sort_pooling/map/while/Pad/paddings/1_1?
)model/sort_pooling/map/while/Pad/paddingsPack4model/sort_pooling/map/while/Pad/paddings/0:output:06model/sort_pooling/map/while/Pad/paddings/1_1:output:0*
N*
T0*
_output_shapes

:2+
)model/sort_pooling/map/while/Pad/paddings?
 model/sort_pooling/map/while/PadPad.model/sort_pooling/map/while/GatherV2:output:02model/sort_pooling/map/while/Pad/paddings:output:0*
T0*'
_output_shapes
:?????????a2"
 model/sort_pooling/map/while/Pad?
Amodel/sort_pooling/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*model_sort_pooling_map_while_placeholder_1(model_sort_pooling_map_while_placeholder)model/sort_pooling/map/while/Pad:output:0*
_output_shapes
: *
element_dtype02C
Amodel/sort_pooling/map/while/TensorArrayV2Write/TensorListSetItem?
"model/sort_pooling/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/sort_pooling/map/while/add/y?
 model/sort_pooling/map/while/addAddV2(model_sort_pooling_map_while_placeholder+model/sort_pooling/map/while/add/y:output:0*
T0*
_output_shapes
: 2"
 model/sort_pooling/map/while/add?
$model/sort_pooling/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/sort_pooling/map/while/add_1/y?
"model/sort_pooling/map/while/add_1AddV2Fmodel_sort_pooling_map_while_model_sort_pooling_map_while_loop_counter-model/sort_pooling/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2$
"model/sort_pooling/map/while/add_1?
%model/sort_pooling/map/while/IdentityIdentity&model/sort_pooling/map/while/add_1:z:0*
T0*
_output_shapes
: 2'
%model/sort_pooling/map/while/Identity?
'model/sort_pooling/map/while/Identity_1IdentityAmodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2)
'model/sort_pooling/map/while/Identity_1?
'model/sort_pooling/map/while/Identity_2Identity$model/sort_pooling/map/while/add:z:0*
T0*
_output_shapes
: 2)
'model/sort_pooling/map/while/Identity_2?
'model/sort_pooling/map/while/Identity_3IdentityQmodel/sort_pooling/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2)
'model/sort_pooling/map/while/Identity_3"W
%model_sort_pooling_map_while_identity.model/sort_pooling/map/while/Identity:output:0"[
'model_sort_pooling_map_while_identity_10model/sort_pooling/map/while/Identity_1:output:0"[
'model_sort_pooling_map_while_identity_20model/sort_pooling/map/while/Identity_2:output:0"[
'model_sort_pooling_map_while_identity_30model/sort_pooling/map/while/Identity_3:output:0"?
Cmodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice_1Emodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice_1_0"?
?model_sort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor?model_sort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
model_sort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor?model_sort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_model_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
 __inference__traced_restore_9806
file_prefix-
)assignvariableop_graph_convolution_kernel1
-assignvariableop_1_graph_convolution_1_kernel1
-assignvariableop_2_graph_convolution_2_kernel1
-assignvariableop_3_graph_convolution_3_kernel$
 assignvariableop_4_conv1d_kernel"
assignvariableop_5_conv1d_bias&
"assignvariableop_6_conv1d_1_kernel$
 assignvariableop_7_conv1d_1_bias#
assignvariableop_8_dense_kernel!
assignvariableop_9_dense_bias&
"assignvariableop_10_dense_1_kernel$
 assignvariableop_11_dense_1_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_17
3assignvariableop_21_adam_graph_convolution_kernel_m9
5assignvariableop_22_adam_graph_convolution_1_kernel_m9
5assignvariableop_23_adam_graph_convolution_2_kernel_m9
5assignvariableop_24_adam_graph_convolution_3_kernel_m,
(assignvariableop_25_adam_conv1d_kernel_m*
&assignvariableop_26_adam_conv1d_bias_m.
*assignvariableop_27_adam_conv1d_1_kernel_m,
(assignvariableop_28_adam_conv1d_1_bias_m+
'assignvariableop_29_adam_dense_kernel_m)
%assignvariableop_30_adam_dense_bias_m-
)assignvariableop_31_adam_dense_1_kernel_m+
'assignvariableop_32_adam_dense_1_bias_m7
3assignvariableop_33_adam_graph_convolution_kernel_v9
5assignvariableop_34_adam_graph_convolution_1_kernel_v9
5assignvariableop_35_adam_graph_convolution_2_kernel_v9
5assignvariableop_36_adam_graph_convolution_3_kernel_v,
(assignvariableop_37_adam_conv1d_kernel_v*
&assignvariableop_38_adam_conv1d_bias_v.
*assignvariableop_39_adam_conv1d_1_kernel_v,
(assignvariableop_40_adam_conv1d_1_bias_v+
'assignvariableop_41_adam_dense_kernel_v)
%assignvariableop_42_adam_dense_bias_v-
)assignvariableop_43_adam_dense_1_kernel_v+
'assignvariableop_44_adam_dense_1_bias_v
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_graph_convolution_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_graph_convolution_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_graph_convolution_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_graph_convolution_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv1d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv1d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_graph_convolution_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_graph_convolution_1_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_graph_convolution_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_graph_convolution_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv1d_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv1d_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_graph_convolution_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_graph_convolution_1_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_graph_convolution_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_graph_convolution_3_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv1d_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv1d_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_8962

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_7375

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_7634

inputs
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpr
MatMulBatchMatMulV2inputs_1inputs*
T0*4
_output_shapes"
 :?????????????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_2_layer_call_fn_9103

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_75222
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_7970

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_9040

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_74512
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
|
'__inference_conv1d_1_layer_call_fn_9423

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_78722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_graph_convolution_2_layer_call_fn_9139
inputs_0
inputs_1
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_75632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_9025

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_9390

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:a*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:a2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
a2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"model_sort_pooling_cond_false_7263'
#model_sort_pooling_cond_placeholderc
_model_sort_pooling_cond_strided_slice_model_sort_pooling_map_tensorarrayv2stack_tensorliststack$
 model_sort_pooling_cond_identity?
+model/sort_pooling/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2-
+model/sort_pooling/cond/strided_slice/stack?
-model/sort_pooling/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2/
-model/sort_pooling/cond/strided_slice/stack_1?
-model/sort_pooling/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2/
-model/sort_pooling/cond/strided_slice/stack_2?
%model/sort_pooling/cond/strided_sliceStridedSlice_model_sort_pooling_cond_strided_slice_model_sort_pooling_map_tensorarrayv2stack_tensorliststack4model/sort_pooling/cond/strided_slice/stack:output:06model/sort_pooling/cond/strided_slice/stack_1:output:06model/sort_pooling/cond/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????a*

begin_mask*
end_mask2'
%model/sort_pooling/cond/strided_slice?
 model/sort_pooling/cond/IdentityIdentity.model/sort_pooling/cond/strided_slice:output:0*
T0*4
_output_shapes"
 :??????????????????a2"
 model/sort_pooling/cond/Identity"M
 model_sort_pooling_cond_identity)model/sort_pooling/cond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?	
y
cond_false_7780
cond_placeholder=
9cond_strided_slice_map_tensorarrayv2stack_tensorliststack
cond_identity?
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
cond/strided_slice/stack?
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2
cond/strided_slice/stack_1?
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
cond/strided_slice/stack_2?
cond/strided_sliceStridedSlice9cond_strided_slice_map_tensorarrayv2stack_tensorliststack!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????a*

begin_mask*
end_mask2
cond/strided_slice?
cond/IdentityIdentitycond/strided_slice:output:0*
T0*4
_output_shapes"
 :??????????????????a2
cond/Identity"'
cond_identitycond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?Q
?
?__inference_model_layer_call_and_return_conditional_losses_7987
input_1
input_2

input_3
graph_convolution_7431
graph_convolution_1_7502
graph_convolution_2_7573
graph_convolution_3_7644
conv1d_7851
conv1d_7853
conv1d_1_7883
conv1d_1_7885

dense_7924

dense_7926
dense_1_7981
dense_1_7983
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?+graph_convolution_2/StatefulPartitionedCall?+graph_convolution_3/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_73752!
dropout/StatefulPartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0input_3graph_convolution_7431*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_74212+
)graph_convolution/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_74462#
!dropout_1/StatefulPartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0input_3graph_convolution_1_7502*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_74922-
+graph_convolution_1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_75172#
!dropout_2/StatefulPartitionedCall?
+graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0input_3graph_convolution_2_7573*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_75632-
+graph_convolution_2/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall4graph_convolution_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_75882#
!dropout_3/StatefulPartitionedCall?
+graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0input_3graph_convolution_3_7644*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_76342-
+graph_convolution_3/StatefulPartitionedCally
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat/concat/axis?
tf.concat/concatConcatV22graph_convolution/StatefulPartitionedCall:output:04graph_convolution_1/StatefulPartitionedCall:output:04graph_convolution_2/StatefulPartitionedCall:output:04graph_convolution_3/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
tf.concat/concat?
sort_pooling/PartitionedCallPartitionedCalltf.concat/concat:output:0input_2*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sort_pooling_layer_call_and_return_conditional_losses_78162
sort_pooling/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall%sort_pooling/PartitionedCall:output:0conv1d_7851conv1d_7853*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_78402 
conv1d/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_73512
max_pooling1d/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_7883conv1d_1_7885*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_78722"
 conv1d_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_78942
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_7924
dense_7926*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_79132
dense/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_79412#
!dropout_4/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_1_7981dense_1_7983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_79702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall,^graph_convolution_2/StatefulPartitionedCall,^graph_convolution_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall2Z
+graph_convolution_2/StatefulPartitionedCall+graph_convolution_2/StatefulPartitionedCall2Z
+graph_convolution_3/StatefulPartitionedCall+graph_convolution_3/StatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?I
?
?__inference_model_layer_call_and_return_conditional_losses_8167

inputs
inputs_1

inputs_2
graph_convolution_8125
graph_convolution_1_8129
graph_convolution_2_8133
graph_convolution_3_8137
conv1d_8143
conv1d_8145
conv1d_1_8149
conv1d_1_8151

dense_8155

dense_8157
dense_1_8161
dense_1_8163
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?+graph_convolution_2/StatefulPartitionedCall?+graph_convolution_3/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_73802
dropout/PartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0inputs_2graph_convolution_8125*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_74212+
)graph_convolution/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_74512
dropout_1/PartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0inputs_2graph_convolution_1_8129*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_74922-
+graph_convolution_1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_75222
dropout_2/PartitionedCall?
+graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0inputs_2graph_convolution_2_8133*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_75632-
+graph_convolution_2/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall4graph_convolution_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_75932
dropout_3/PartitionedCall?
+graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0inputs_2graph_convolution_3_8137*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_76342-
+graph_convolution_3/StatefulPartitionedCally
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat/concat/axis?
tf.concat/concatConcatV22graph_convolution/StatefulPartitionedCall:output:04graph_convolution_1/StatefulPartitionedCall:output:04graph_convolution_2/StatefulPartitionedCall:output:04graph_convolution_3/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
tf.concat/concat?
sort_pooling/PartitionedCallPartitionedCalltf.concat/concat:output:0inputs_1*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sort_pooling_layer_call_and_return_conditional_losses_78162
sort_pooling/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall%sort_pooling/PartitionedCall:output:0conv1d_8143conv1d_8145*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_78402 
conv1d/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_73512
max_pooling1d/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_8149conv1d_1_8151*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_78722"
 conv1d_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_78942
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_8155
dense_8157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_79132
dense/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_79462
dropout_4/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_1_8161dense_1_8163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_79702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall,^graph_convolution_2/StatefulPartitionedCall,^graph_convolution_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall2Z
+graph_convolution_2/StatefulPartitionedCall+graph_convolution_2/StatefulPartitionedCall2Z
+graph_convolution_3/StatefulPartitionedCall+graph_convolution_3/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_8977

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_73802
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
y
cond_false_9333
cond_placeholder=
9cond_strided_slice_map_tensorarrayv2stack_tensorliststack
cond_identity?
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
cond/strided_slice/stack?
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2
cond/strided_slice/stack_1?
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
cond/strided_slice/stack_2?
cond/strided_sliceStridedSlice9cond_strided_slice_map_tensorarrayv2stack_tensorliststack!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????a*

begin_mask*
end_mask2
cond/strided_slice?
cond/IdentityIdentitycond/strided_slice:output:0*
T0*4
_output_shapes"
 :??????????????????a2
cond/Identity"'
cond_identitycond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?
?
$__inference_model_layer_call_fn_8194
input_1
input_2

input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_81672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
?
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_9194
inputs_0
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpt
MatMulBatchMatMulV2inputs_1inputs_0*
T0*4
_output_shapes"
 :?????????????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_7894

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_9088

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?	
?
map_while_cond_9222$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_9222___redundant_placeholder0:
6map_while_map_while_cond_9222___redundant_placeholder1
map_while_identity
?
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less?
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
??
?
 sort_pooling_map_while_body_8383>
:sort_pooling_map_while_sort_pooling_map_while_loop_counter9
5sort_pooling_map_while_sort_pooling_map_strided_slice&
"sort_pooling_map_while_placeholder(
$sort_pooling_map_while_placeholder_1=
9sort_pooling_map_while_sort_pooling_map_strided_slice_1_0y
usort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0}
ysort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0#
sort_pooling_map_while_identity%
!sort_pooling_map_while_identity_1%
!sort_pooling_map_while_identity_2%
!sort_pooling_map_while_identity_3;
7sort_pooling_map_while_sort_pooling_map_strided_slice_1w
ssort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor{
wsort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor?
Hsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2J
Hsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shape?
:sort_pooling/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0"sort_pooling_map_while_placeholderQsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????a*
element_dtype02<
:sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem?
Jsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2L
Jsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
<sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemysort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0"sort_pooling_map_while_placeholderSsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*#
_output_shapes
:?????????*
element_dtype0
2>
<sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem?
)sort_pooling/map/while/boolean_mask/ShapeShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2+
)sort_pooling/map/while/boolean_mask/Shape?
7sort_pooling/map/while/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sort_pooling/map/while/boolean_mask/strided_slice/stack?
9sort_pooling/map/while/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sort_pooling/map/while/boolean_mask/strided_slice/stack_1?
9sort_pooling/map/while/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sort_pooling/map/while/boolean_mask/strided_slice/stack_2?
1sort_pooling/map/while/boolean_mask/strided_sliceStridedSlice2sort_pooling/map/while/boolean_mask/Shape:output:0@sort_pooling/map/while/boolean_mask/strided_slice/stack:output:0Bsort_pooling/map/while/boolean_mask/strided_slice/stack_1:output:0Bsort_pooling/map/while/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1sort_pooling/map/while/boolean_mask/strided_slice?
:sort_pooling/map/while/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sort_pooling/map/while/boolean_mask/Prod/reduction_indices?
(sort_pooling/map/while/boolean_mask/ProdProd:sort_pooling/map/while/boolean_mask/strided_slice:output:0Csort_pooling/map/while/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2*
(sort_pooling/map/while/boolean_mask/Prod?
+sort_pooling/map/while/boolean_mask/Shape_1ShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2-
+sort_pooling/map/while/boolean_mask/Shape_1?
9sort_pooling/map/while/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9sort_pooling/map/while/boolean_mask/strided_slice_1/stack?
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1?
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2?
3sort_pooling/map/while/boolean_mask/strided_slice_1StridedSlice4sort_pooling/map/while/boolean_mask/Shape_1:output:0Bsort_pooling/map/while/boolean_mask/strided_slice_1/stack:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_1/stack_1:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask25
3sort_pooling/map/while/boolean_mask/strided_slice_1?
+sort_pooling/map/while/boolean_mask/Shape_2ShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2-
+sort_pooling/map/while/boolean_mask/Shape_2?
9sort_pooling/map/while/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9sort_pooling/map/while/boolean_mask/strided_slice_2/stack?
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1?
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2?
3sort_pooling/map/while/boolean_mask/strided_slice_2StridedSlice4sort_pooling/map/while/boolean_mask/Shape_2:output:0Bsort_pooling/map/while/boolean_mask/strided_slice_2/stack:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_2/stack_1:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3sort_pooling/map/while/boolean_mask/strided_slice_2?
3sort_pooling/map/while/boolean_mask/concat/values_1Pack1sort_pooling/map/while/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:25
3sort_pooling/map/while/boolean_mask/concat/values_1?
/sort_pooling/map/while/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sort_pooling/map/while/boolean_mask/concat/axis?
*sort_pooling/map/while/boolean_mask/concatConcatV2<sort_pooling/map/while/boolean_mask/strided_slice_1:output:0<sort_pooling/map/while/boolean_mask/concat/values_1:output:0<sort_pooling/map/while/boolean_mask/strided_slice_2:output:08sort_pooling/map/while/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*sort_pooling/map/while/boolean_mask/concat?
+sort_pooling/map/while/boolean_mask/ReshapeReshapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:03sort_pooling/map/while/boolean_mask/concat:output:0*
T0*'
_output_shapes
:?????????a2-
+sort_pooling/map/while/boolean_mask/Reshape?
3sort_pooling/map/while/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3sort_pooling/map/while/boolean_mask/Reshape_1/shape?
-sort_pooling/map/while/boolean_mask/Reshape_1ReshapeCsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem:item:0<sort_pooling/map/while/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2/
-sort_pooling/map/while/boolean_mask/Reshape_1?
)sort_pooling/map/while/boolean_mask/WhereWhere6sort_pooling/map/while/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2+
)sort_pooling/map/while/boolean_mask/Where?
+sort_pooling/map/while/boolean_mask/SqueezeSqueeze1sort_pooling/map/while/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2-
+sort_pooling/map/while/boolean_mask/Squeeze?
1sort_pooling/map/while/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sort_pooling/map/while/boolean_mask/GatherV2/axis?
,sort_pooling/map/while/boolean_mask/GatherV2GatherV24sort_pooling/map/while/boolean_mask/Reshape:output:04sort_pooling/map/while/boolean_mask/Squeeze:output:0:sort_pooling/map/while/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:?????????a2.
,sort_pooling/map/while/boolean_mask/GatherV2?
*sort_pooling/map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2,
*sort_pooling/map/while/strided_slice/stack?
,sort_pooling/map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,sort_pooling/map/while/strided_slice/stack_1?
,sort_pooling/map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,sort_pooling/map/while/strided_slice/stack_2?
$sort_pooling/map/while/strided_sliceStridedSlice5sort_pooling/map/while/boolean_mask/GatherV2:output:03sort_pooling/map/while/strided_slice/stack:output:05sort_pooling/map/while/strided_slice/stack_1:output:05sort_pooling/map/while/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2&
$sort_pooling/map/while/strided_slice?
#sort_pooling/map/while/argsort/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sort_pooling/map/while/argsort/axis?
$sort_pooling/map/while/argsort/ShapeShape-sort_pooling/map/while/strided_slice:output:0*
T0*
_output_shapes
:2&
$sort_pooling/map/while/argsort/Shape?
2sort_pooling/map/while/argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sort_pooling/map/while/argsort/strided_slice/stack?
4sort_pooling/map/while/argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sort_pooling/map/while/argsort/strided_slice/stack_1?
4sort_pooling/map/while/argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sort_pooling/map/while/argsort/strided_slice/stack_2?
,sort_pooling/map/while/argsort/strided_sliceStridedSlice-sort_pooling/map/while/argsort/Shape:output:0;sort_pooling/map/while/argsort/strided_slice/stack:output:0=sort_pooling/map/while/argsort/strided_slice/stack_1:output:0=sort_pooling/map/while/argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sort_pooling/map/while/argsort/strided_slice?
#sort_pooling/map/while/argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#sort_pooling/map/while/argsort/Rank?
%sort_pooling/map/while/argsort/TopKV2TopKV2-sort_pooling/map/while/strided_slice:output:05sort_pooling/map/while/argsort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????2'
%sort_pooling/map/while/argsort/TopKV2?
$sort_pooling/map/while/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$sort_pooling/map/while/GatherV2/axis?
sort_pooling/map/while/GatherV2GatherV2Asort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0/sort_pooling/map/while/argsort/TopKV2:indices:0-sort_pooling/map/while/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????a2!
sort_pooling/map/while/GatherV2?
sort_pooling/map/while/ShapeShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
sort_pooling/map/while/Shape?
,sort_pooling/map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sort_pooling/map/while/strided_slice_1/stack?
.sort_pooling/map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_1/stack_1?
.sort_pooling/map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_1/stack_2?
&sort_pooling/map/while/strided_slice_1StridedSlice%sort_pooling/map/while/Shape:output:05sort_pooling/map/while/strided_slice_1/stack:output:07sort_pooling/map/while/strided_slice_1/stack_1:output:07sort_pooling/map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sort_pooling/map/while/strided_slice_1?
sort_pooling/map/while/Shape_1Shape(sort_pooling/map/while/GatherV2:output:0*
T0*
_output_shapes
:2 
sort_pooling/map/while/Shape_1?
,sort_pooling/map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sort_pooling/map/while/strided_slice_2/stack?
.sort_pooling/map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_2/stack_1?
.sort_pooling/map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_2/stack_2?
&sort_pooling/map/while/strided_slice_2StridedSlice'sort_pooling/map/while/Shape_1:output:05sort_pooling/map/while/strided_slice_2/stack:output:07sort_pooling/map/while/strided_slice_2/stack_1:output:07sort_pooling/map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sort_pooling/map/while/strided_slice_2?
sort_pooling/map/while/subSub/sort_pooling/map/while/strided_slice_1:output:0/sort_pooling/map/while/strided_slice_2:output:0*
T0*
_output_shapes
: 2
sort_pooling/map/while/sub?
'sort_pooling/map/while/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sort_pooling/map/while/Pad/paddings/0/0?
%sort_pooling/map/while/Pad/paddings/0Pack0sort_pooling/map/while/Pad/paddings/0/0:output:0sort_pooling/map/while/sub:z:0*
N*
T0*
_output_shapes
:2'
%sort_pooling/map/while/Pad/paddings/0?
'sort_pooling/map/while/Pad/paddings/1_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'sort_pooling/map/while/Pad/paddings/1_1?
#sort_pooling/map/while/Pad/paddingsPack.sort_pooling/map/while/Pad/paddings/0:output:00sort_pooling/map/while/Pad/paddings/1_1:output:0*
N*
T0*
_output_shapes

:2%
#sort_pooling/map/while/Pad/paddings?
sort_pooling/map/while/PadPad(sort_pooling/map/while/GatherV2:output:0,sort_pooling/map/while/Pad/paddings:output:0*
T0*'
_output_shapes
:?????????a2
sort_pooling/map/while/Pad?
;sort_pooling/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sort_pooling_map_while_placeholder_1"sort_pooling_map_while_placeholder#sort_pooling/map/while/Pad:output:0*
_output_shapes
: *
element_dtype02=
;sort_pooling/map/while/TensorArrayV2Write/TensorListSetItem~
sort_pooling/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sort_pooling/map/while/add/y?
sort_pooling/map/while/addAddV2"sort_pooling_map_while_placeholder%sort_pooling/map/while/add/y:output:0*
T0*
_output_shapes
: 2
sort_pooling/map/while/add?
sort_pooling/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sort_pooling/map/while/add_1/y?
sort_pooling/map/while/add_1AddV2:sort_pooling_map_while_sort_pooling_map_while_loop_counter'sort_pooling/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sort_pooling/map/while/add_1?
sort_pooling/map/while/IdentityIdentity sort_pooling/map/while/add_1:z:0*
T0*
_output_shapes
: 2!
sort_pooling/map/while/Identity?
!sort_pooling/map/while/Identity_1Identity5sort_pooling_map_while_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2#
!sort_pooling/map/while/Identity_1?
!sort_pooling/map/while/Identity_2Identitysort_pooling/map/while/add:z:0*
T0*
_output_shapes
: 2#
!sort_pooling/map/while/Identity_2?
!sort_pooling/map/while/Identity_3IdentityKsort_pooling/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2#
!sort_pooling/map/while/Identity_3"K
sort_pooling_map_while_identity(sort_pooling/map/while/Identity:output:0"O
!sort_pooling_map_while_identity_1*sort_pooling/map/while/Identity_1:output:0"O
!sort_pooling_map_while_identity_2*sort_pooling/map/while/Identity_2:output:0"O
!sort_pooling_map_while_identity_3*sort_pooling/map/while/Identity_3:output:0"t
7sort_pooling_map_while_sort_pooling_map_strided_slice_19sort_pooling_map_while_sort_pooling_map_strided_slice_1_0"?
wsort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensorysort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
ssort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensorusort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?	
?__inference_model_layer_call_and_return_conditional_losses_8888
inputs_0
inputs_1

inputs_25
1graph_convolution_shape_2_readvariableop_resource7
3graph_convolution_1_shape_2_readvariableop_resource7
3graph_convolution_2_shape_2_readvariableop_resource7
3graph_convolution_3_shape_2_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?*graph_convolution/transpose/ReadVariableOp?,graph_convolution_1/transpose/ReadVariableOp?,graph_convolution_2/transpose/ReadVariableOp?,graph_convolution_3/transpose/ReadVariableOpy
dropout/IdentityIdentityinputs_0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Identity?
graph_convolution/MatMulBatchMatMulV2inputs_2dropout/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_convolution/MatMul?
graph_convolution/ShapeShape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape?
graph_convolution/Shape_1Shape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape_1?
graph_convolution/unstackUnpack"graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution/unstack?
(graph_convolution/Shape_2/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02*
(graph_convolution/Shape_2/ReadVariableOp?
graph_convolution/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_convolution/Shape_2?
graph_convolution/unstack_1Unpack"graph_convolution/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution/unstack_1?
graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2!
graph_convolution/Reshape/shape?
graph_convolution/ReshapeReshape!graph_convolution/MatMul:output:0(graph_convolution/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
graph_convolution/Reshape?
*graph_convolution/transpose/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02,
*graph_convolution/transpose/ReadVariableOp?
 graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2"
 graph_convolution/transpose/perm?
graph_convolution/transpose	Transpose2graph_convolution/transpose/ReadVariableOp:value:0)graph_convolution/transpose/perm:output:0*
T0*
_output_shapes

: 2
graph_convolution/transpose?
!graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2#
!graph_convolution/Reshape_1/shape?
graph_convolution/Reshape_1Reshapegraph_convolution/transpose:y:0*graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
graph_convolution/Reshape_1?
graph_convolution/MatMul_1MatMul"graph_convolution/Reshape:output:0$graph_convolution/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution/MatMul_1?
#graph_convolution/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#graph_convolution/Reshape_2/shape/2?
!graph_convolution/Reshape_2/shapePack"graph_convolution/unstack:output:0"graph_convolution/unstack:output:1,graph_convolution/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!graph_convolution/Reshape_2/shape?
graph_convolution/Reshape_2Reshape$graph_convolution/MatMul_1:product:0*graph_convolution/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution/Reshape_2?
graph_convolution/TanhTanh$graph_convolution/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution/Tanh?
dropout_1/IdentityIdentitygraph_convolution/Tanh:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_1/Identity?
graph_convolution_1/MatMulBatchMatMulV2inputs_2dropout_1/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_1/MatMul?
graph_convolution_1/ShapeShape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape?
graph_convolution_1/Shape_1Shape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape_1?
graph_convolution_1/unstackUnpack$graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_1/unstack?
*graph_convolution_1/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02,
*graph_convolution_1/Shape_2/ReadVariableOp?
graph_convolution_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2
graph_convolution_1/Shape_2?
graph_convolution_1/unstack_1Unpack$graph_convolution_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_1/unstack_1?
!graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!graph_convolution_1/Reshape/shape?
graph_convolution_1/ReshapeReshape#graph_convolution_1/MatMul:output:0*graph_convolution_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_1/Reshape?
,graph_convolution_1/transpose/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02.
,graph_convolution_1/transpose/ReadVariableOp?
"graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_1/transpose/perm?
graph_convolution_1/transpose	Transpose4graph_convolution_1/transpose/ReadVariableOp:value:0+graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
graph_convolution_1/transpose?
#graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2%
#graph_convolution_1/Reshape_1/shape?
graph_convolution_1/Reshape_1Reshape!graph_convolution_1/transpose:y:0,graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
graph_convolution_1/Reshape_1?
graph_convolution_1/MatMul_1MatMul$graph_convolution_1/Reshape:output:0&graph_convolution_1/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_1/MatMul_1?
%graph_convolution_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2'
%graph_convolution_1/Reshape_2/shape/2?
#graph_convolution_1/Reshape_2/shapePack$graph_convolution_1/unstack:output:0$graph_convolution_1/unstack:output:1.graph_convolution_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_1/Reshape_2/shape?
graph_convolution_1/Reshape_2Reshape&graph_convolution_1/MatMul_1:product:0,graph_convolution_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_1/Reshape_2?
graph_convolution_1/TanhTanh&graph_convolution_1/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_1/Tanh?
dropout_2/IdentityIdentitygraph_convolution_1/Tanh:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_2/Identity?
graph_convolution_2/MatMulBatchMatMulV2inputs_2dropout_2/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_2/MatMul?
graph_convolution_2/ShapeShape#graph_convolution_2/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_2/Shape?
graph_convolution_2/Shape_1Shape#graph_convolution_2/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_2/Shape_1?
graph_convolution_2/unstackUnpack$graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_2/unstack?
*graph_convolution_2/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_2_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02,
*graph_convolution_2/Shape_2/ReadVariableOp?
graph_convolution_2/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2
graph_convolution_2/Shape_2?
graph_convolution_2/unstack_1Unpack$graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_2/unstack_1?
!graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!graph_convolution_2/Reshape/shape?
graph_convolution_2/ReshapeReshape#graph_convolution_2/MatMul:output:0*graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_2/Reshape?
,graph_convolution_2/transpose/ReadVariableOpReadVariableOp3graph_convolution_2_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02.
,graph_convolution_2/transpose/ReadVariableOp?
"graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_2/transpose/perm?
graph_convolution_2/transpose	Transpose4graph_convolution_2/transpose/ReadVariableOp:value:0+graph_convolution_2/transpose/perm:output:0*
T0*
_output_shapes

:  2
graph_convolution_2/transpose?
#graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2%
#graph_convolution_2/Reshape_1/shape?
graph_convolution_2/Reshape_1Reshape!graph_convolution_2/transpose:y:0,graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
graph_convolution_2/Reshape_1?
graph_convolution_2/MatMul_1MatMul$graph_convolution_2/Reshape:output:0&graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_2/MatMul_1?
%graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2'
%graph_convolution_2/Reshape_2/shape/2?
#graph_convolution_2/Reshape_2/shapePack$graph_convolution_2/unstack:output:0$graph_convolution_2/unstack:output:1.graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_2/Reshape_2/shape?
graph_convolution_2/Reshape_2Reshape&graph_convolution_2/MatMul_1:product:0,graph_convolution_2/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_2/Reshape_2?
graph_convolution_2/TanhTanh&graph_convolution_2/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_2/Tanh?
dropout_3/IdentityIdentitygraph_convolution_2/Tanh:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_3/Identity?
graph_convolution_3/MatMulBatchMatMulV2inputs_2dropout_3/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_3/MatMul?
graph_convolution_3/ShapeShape#graph_convolution_3/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_3/Shape?
graph_convolution_3/Shape_1Shape#graph_convolution_3/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_3/Shape_1?
graph_convolution_3/unstackUnpack$graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_3/unstack?
*graph_convolution_3/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_3_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02,
*graph_convolution_3/Shape_2/ReadVariableOp?
graph_convolution_3/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_convolution_3/Shape_2?
graph_convolution_3/unstack_1Unpack$graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_3/unstack_1?
!graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!graph_convolution_3/Reshape/shape?
graph_convolution_3/ReshapeReshape#graph_convolution_3/MatMul:output:0*graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_3/Reshape?
,graph_convolution_3/transpose/ReadVariableOpReadVariableOp3graph_convolution_3_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02.
,graph_convolution_3/transpose/ReadVariableOp?
"graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_3/transpose/perm?
graph_convolution_3/transpose	Transpose4graph_convolution_3/transpose/ReadVariableOp:value:0+graph_convolution_3/transpose/perm:output:0*
T0*
_output_shapes

: 2
graph_convolution_3/transpose?
#graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2%
#graph_convolution_3/Reshape_1/shape?
graph_convolution_3/Reshape_1Reshape!graph_convolution_3/transpose:y:0,graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
graph_convolution_3/Reshape_1?
graph_convolution_3/MatMul_1MatMul$graph_convolution_3/Reshape:output:0&graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
graph_convolution_3/MatMul_1?
%graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%graph_convolution_3/Reshape_2/shape/2?
#graph_convolution_3/Reshape_2/shapePack$graph_convolution_3/unstack:output:0$graph_convolution_3/unstack:output:1.graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_3/Reshape_2/shape?
graph_convolution_3/Reshape_2Reshape&graph_convolution_3/MatMul_1:product:0,graph_convolution_3/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_convolution_3/Reshape_2?
graph_convolution_3/TanhTanh&graph_convolution_3/Reshape_2:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_convolution_3/Tanhy
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat/concat/axis?
tf.concat/concatConcatV2graph_convolution/Tanh:y:0graph_convolution_1/Tanh:y:0graph_convolution_2/Tanh:y:0graph_convolution_3/Tanh:y:0tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
tf.concat/concaty
sort_pooling/map/ShapeShapetf.concat/concat:output:0*
T0*
_output_shapes
:2
sort_pooling/map/Shape?
$sort_pooling/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sort_pooling/map/strided_slice/stack?
&sort_pooling/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sort_pooling/map/strided_slice/stack_1?
&sort_pooling/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sort_pooling/map/strided_slice/stack_2?
sort_pooling/map/strided_sliceStridedSlicesort_pooling/map/Shape:output:0-sort_pooling/map/strided_slice/stack:output:0/sort_pooling/map/strided_slice/stack_1:output:0/sort_pooling/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sort_pooling/map/strided_slice?
,sort_pooling/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sort_pooling/map/TensorArrayV2/element_shape?
sort_pooling/map/TensorArrayV2TensorListReserve5sort_pooling/map/TensorArrayV2/element_shape:output:0'sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sort_pooling/map/TensorArrayV2?
.sort_pooling/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sort_pooling/map/TensorArrayV2_1/element_shape?
 sort_pooling/map/TensorArrayV2_1TensorListReserve7sort_pooling/map/TensorArrayV2_1/element_shape:output:0'sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02"
 sort_pooling/map/TensorArrayV2_1?
Fsort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2H
Fsort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
8sort_pooling/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensortf.concat/concat:output:0Osort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8sort_pooling/map/TensorArrayUnstack/TensorListFromTensor?
Hsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2J
Hsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
:sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_1Qsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02<
:sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensorr
sort_pooling/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
sort_pooling/map/Const?
.sort_pooling/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sort_pooling/map/TensorArrayV2_2/element_shape?
 sort_pooling/map/TensorArrayV2_2TensorListReserve7sort_pooling/map/TensorArrayV2_2/element_shape:output:0'sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sort_pooling/map/TensorArrayV2_2?
#sort_pooling/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sort_pooling/map/while/loop_counter?
sort_pooling/map/whileStatelessWhile,sort_pooling/map/while/loop_counter:output:0'sort_pooling/map/strided_slice:output:0sort_pooling/map/Const:output:0)sort_pooling/map/TensorArrayV2_2:handle:0'sort_pooling/map/strided_slice:output:0Hsort_pooling/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *,
body$R"
 sort_pooling_map_while_body_8699*,
cond$R"
 sort_pooling_map_while_cond_8698*!
output_shapes
: : : : : : : 2
sort_pooling/map/while?
Asort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2C
Asort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shape?
3sort_pooling/map/TensorArrayV2Stack/TensorListStackTensorListStacksort_pooling/map/while:output:3Jsort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????a*
element_dtype025
3sort_pooling/map/TensorArrayV2Stack/TensorListStack?
sort_pooling/ShapeShape<sort_pooling/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
sort_pooling/Shapel
sort_pooling/Less/yConst*
_output_shapes
: *
dtype0*
value	B :#2
sort_pooling/Less/y?
sort_pooling/LessLesssort_pooling/Shape:output:0sort_pooling/Less/y:output:0*
T0*
_output_shapes
:2
sort_pooling/Less?
 sort_pooling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sort_pooling/strided_slice/stack?
"sort_pooling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sort_pooling/strided_slice/stack_1?
"sort_pooling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sort_pooling/strided_slice/stack_2?
sort_pooling/strided_sliceStridedSlicesort_pooling/Less:z:0)sort_pooling/strided_slice/stack:output:0+sort_pooling/strided_slice/stack_1:output:0+sort_pooling/strided_slice/stack_2:output:0*
Index0*
T0
*
_output_shapes
: *
shrink_axis_mask2
sort_pooling/strided_slice?
sort_pooling/condStatelessIf#sort_pooling/strided_slice:output:0sort_pooling/Shape:output:0<sort_pooling/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :??????????????????a* 
_read_only_resource_inputs
 */
else_branch R
sort_pooling_cond_false_8809*3
output_shapes"
 :??????????????????a*.
then_branchR
sort_pooling_cond_true_88082
sort_pooling/cond?
sort_pooling/cond/IdentityIdentitysort_pooling/cond:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Identity?
"sort_pooling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sort_pooling/strided_slice_1/stack?
$sort_pooling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sort_pooling/strided_slice_1/stack_1?
$sort_pooling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sort_pooling/strided_slice_1/stack_2?
sort_pooling/strided_slice_1StridedSlicesort_pooling/Shape:output:0+sort_pooling/strided_slice_1/stack:output:0-sort_pooling/strided_slice_1/stack_1:output:0-sort_pooling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sort_pooling/strided_slice_1
sort_pooling/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
sort_pooling/Reshape/shape/1~
sort_pooling/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
sort_pooling/Reshape/shape/2?
sort_pooling/Reshape/shapePack%sort_pooling/strided_slice_1:output:0%sort_pooling/Reshape/shape/1:output:0%sort_pooling/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
sort_pooling/Reshape/shape?
sort_pooling/ReshapeReshape#sort_pooling/cond/Identity:output:0#sort_pooling/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
sort_pooling/Reshape?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimssort_pooling/Reshape:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:a*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:a2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
a2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2
conv1d/BiasAdd~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d/BiasAdd:output:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2
max_pooling1d/Squeeze?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsmax_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d_1/BiasAddo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv1d_1/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dropout_4/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_4/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_4/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp+^graph_convolution/transpose/ReadVariableOp-^graph_convolution_1/transpose/ReadVariableOp-^graph_convolution_2/transpose/ReadVariableOp-^graph_convolution_3/transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2X
*graph_convolution/transpose/ReadVariableOp*graph_convolution/transpose/ReadVariableOp2\
,graph_convolution_1/transpose/ReadVariableOp,graph_convolution_1/transpose/ReadVariableOp2\
,graph_convolution_2/transpose/ReadVariableOp,graph_convolution_2/transpose/ReadVariableOp2\
,graph_convolution_3/transpose/ReadVariableOp,graph_convolution_3/transpose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?
_
&__inference_dropout_layer_call_fn_8972

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_73752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_7421

inputs
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpr
MatMulBatchMatMulV2inputs_1inputs*
T0*4
_output_shapes"
 :??????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:??????????????????:'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7522

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :?????????????????? 2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_7492

inputs
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpr
MatMulBatchMatMulV2inputs_1inputs*
T0*4
_output_shapes"
 :?????????????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?r
?
map_while_body_7670$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensora
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????a*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItem?
=map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
=map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*#
_output_shapes
:?????????*
element_dtype0
21
/map/while/TensorArrayV2Read_1/TensorListGetItem?
map/while/boolean_mask/ShapeShape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
map/while/boolean_mask/Shape?
*map/while/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*map/while/boolean_mask/strided_slice/stack?
,map/while/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,map/while/boolean_mask/strided_slice/stack_1?
,map/while/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,map/while/boolean_mask/strided_slice/stack_2?
$map/while/boolean_mask/strided_sliceStridedSlice%map/while/boolean_mask/Shape:output:03map/while/boolean_mask/strided_slice/stack:output:05map/while/boolean_mask/strided_slice/stack_1:output:05map/while/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2&
$map/while/boolean_mask/strided_slice?
-map/while/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2/
-map/while/boolean_mask/Prod/reduction_indices?
map/while/boolean_mask/ProdProd-map/while/boolean_mask/strided_slice:output:06map/while/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
map/while/boolean_mask/Prod?
map/while/boolean_mask/Shape_1Shape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
map/while/boolean_mask/Shape_1?
,map/while/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,map/while/boolean_mask/strided_slice_1/stack?
.map/while/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.map/while/boolean_mask/strided_slice_1/stack_1?
.map/while/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.map/while/boolean_mask/strided_slice_1/stack_2?
&map/while/boolean_mask/strided_slice_1StridedSlice'map/while/boolean_mask/Shape_1:output:05map/while/boolean_mask/strided_slice_1/stack:output:07map/while/boolean_mask/strided_slice_1/stack_1:output:07map/while/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2(
&map/while/boolean_mask/strided_slice_1?
map/while/boolean_mask/Shape_2Shape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
map/while/boolean_mask/Shape_2?
,map/while/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,map/while/boolean_mask/strided_slice_2/stack?
.map/while/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.map/while/boolean_mask/strided_slice_2/stack_1?
.map/while/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.map/while/boolean_mask/strided_slice_2/stack_2?
&map/while/boolean_mask/strided_slice_2StridedSlice'map/while/boolean_mask/Shape_2:output:05map/while/boolean_mask/strided_slice_2/stack:output:07map/while/boolean_mask/strided_slice_2/stack_1:output:07map/while/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&map/while/boolean_mask/strided_slice_2?
&map/while/boolean_mask/concat/values_1Pack$map/while/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2(
&map/while/boolean_mask/concat/values_1?
"map/while/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"map/while/boolean_mask/concat/axis?
map/while/boolean_mask/concatConcatV2/map/while/boolean_mask/strided_slice_1:output:0/map/while/boolean_mask/concat/values_1:output:0/map/while/boolean_mask/strided_slice_2:output:0+map/while/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
map/while/boolean_mask/concat?
map/while/boolean_mask/ReshapeReshape4map/while/TensorArrayV2Read/TensorListGetItem:item:0&map/while/boolean_mask/concat:output:0*
T0*'
_output_shapes
:?????????a2 
map/while/boolean_mask/Reshape?
&map/while/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&map/while/boolean_mask/Reshape_1/shape?
 map/while/boolean_mask/Reshape_1Reshape6map/while/TensorArrayV2Read_1/TensorListGetItem:item:0/map/while/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2"
 map/while/boolean_mask/Reshape_1?
map/while/boolean_mask/WhereWhere)map/while/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2
map/while/boolean_mask/Where?
map/while/boolean_mask/SqueezeSqueeze$map/while/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2 
map/while/boolean_mask/Squeeze?
$map/while/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$map/while/boolean_mask/GatherV2/axis?
map/while/boolean_mask/GatherV2GatherV2'map/while/boolean_mask/Reshape:output:0'map/while/boolean_mask/Squeeze:output:0-map/while/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:?????????a2!
map/while/boolean_mask/GatherV2?
map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
map/while/strided_slice/stack?
map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
map/while/strided_slice/stack_1?
map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
map/while/strided_slice/stack_2?
map/while/strided_sliceStridedSlice(map/while/boolean_mask/GatherV2:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
map/while/strided_slicer
map/while/argsort/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/argsort/axis?
map/while/argsort/ShapeShape map/while/strided_slice:output:0*
T0*
_output_shapes
:2
map/while/argsort/Shape?
%map/while/argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%map/while/argsort/strided_slice/stack?
'map/while/argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'map/while/argsort/strided_slice/stack_1?
'map/while/argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'map/while/argsort/strided_slice/stack_2?
map/while/argsort/strided_sliceStridedSlice map/while/argsort/Shape:output:0.map/while/argsort/strided_slice/stack:output:00map/while/argsort/strided_slice/stack_1:output:00map/while/argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
map/while/argsort/strided_slicer
map/while/argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/argsort/Rank?
map/while/argsort/TopKV2TopKV2 map/while/strided_slice:output:0(map/while/argsort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????2
map/while/argsort/TopKV2t
map/while/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/GatherV2/axis?
map/while/GatherV2GatherV24map/while/TensorArrayV2Read/TensorListGetItem:item:0"map/while/argsort/TopKV2:indices:0 map/while/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????a2
map/while/GatherV2?
map/while/ShapeShape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
map/while/Shape?
map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
map/while/strided_slice_1/stack?
!map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_1/stack_1?
!map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_1/stack_2?
map/while/strided_slice_1StridedSlicemap/while/Shape:output:0(map/while/strided_slice_1/stack:output:0*map/while/strided_slice_1/stack_1:output:0*map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/while/strided_slice_1q
map/while/Shape_1Shapemap/while/GatherV2:output:0*
T0*
_output_shapes
:2
map/while/Shape_1?
map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
map/while/strided_slice_2/stack?
!map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_2/stack_1?
!map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_2/stack_2?
map/while/strided_slice_2StridedSlicemap/while/Shape_1:output:0(map/while/strided_slice_2/stack:output:0*map/while/strided_slice_2/stack_1:output:0*map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/while/strided_slice_2?
map/while/subSub"map/while/strided_slice_1:output:0"map/while/strided_slice_2:output:0*
T0*
_output_shapes
: 2
map/while/subz
map/while/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
map/while/Pad/paddings/0/0?
map/while/Pad/paddings/0Pack#map/while/Pad/paddings/0/0:output:0map/while/sub:z:0*
N*
T0*
_output_shapes
:2
map/while/Pad/paddings/0?
map/while/Pad/paddings/1_1Const*
_output_shapes
:*
dtype0*
valueB"        2
map/while/Pad/paddings/1_1?
map/while/Pad/paddingsPack!map/while/Pad/paddings/0:output:0#map/while/Pad/paddings/1_1:output:0*
N*
T0*
_output_shapes

:2
map/while/Pad/paddings?
map/while/PadPadmap/while/GatherV2:output:0map/while/Pad/paddings:output:0*
T0*'
_output_shapes
:?????????a2
map/while/Pad?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/Pad:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/y?
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1j
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: 2
map/while/Identityv
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Identity_1l
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 2
map/while/Identity_2?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
map/while/Identity_3"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"?
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?6
p
F__inference_sort_pooling_layer_call_and_return_conditional_losses_7816

embeddings
mask

identityP
	map/ShapeShape
embeddings*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack?
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1?
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice?
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
map/TensorArrayV2/element_shape?
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2?
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_1/element_shape?
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
map/TensorArrayV2_1?
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shape?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor
embeddingsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensor?
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2=
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
-map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormaskDmap/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02/
-map/TensorArrayUnstack_1/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const?
!map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_2/element_shape?
map/TensorArrayV2_2TensorListReserve*map/TensorArrayV2_2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_2r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counter?
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_2:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *
bodyR
map_while_body_7670*
condR
map_while_cond_7669*!
output_shapes
: : : : : : : 2
	map/while?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   26
4map/TensorArrayV2Stack/TensorListStack/element_shape?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????a*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStackm
ShapeShape/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
ShapeR
Less/yConst*
_output_shapes
: *
dtype0*
value	B :#2
Less/yZ
LessLessShape:output:0Less/y:output:0*
T0*
_output_shapes
:2
Lesst
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceLess:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
condStatelessIfstrided_slice:output:0Shape:output:0/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :??????????????????a* 
_read_only_resource_inputs
 *"
else_branchR
cond_false_7780*3
output_shapes"
 :??????????????????a*!
then_branchR
cond_true_77792
condx
cond/IdentityIdentitycond:output:0*
T0*4
_output_shapes"
 :??????????????????a2
cond/Identityx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1e
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapecond/Identity:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????????????a:??????????????????:` \
4
_output_shapes"
 :??????????????????a
$
_user_specified_name
embeddings:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
$__inference_model_layer_call_fn_8115
input_1
input_2

input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_80882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
H
,__inference_max_pooling1d_layer_call_fn_7357

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_73512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_9156

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :?????????????????? 2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_graph_convolution_3_layer_call_fn_9202
inputs_0
inputs_1
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_76342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_9414

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?6
p
F__inference_sort_pooling_layer_call_and_return_conditional_losses_9369

embeddings
mask

identityP
	map/ShapeShape
embeddings*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack?
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1?
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice?
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
map/TensorArrayV2/element_shape?
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2?
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_1/element_shape?
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
map/TensorArrayV2_1?
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shape?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor
embeddingsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensor?
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2=
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
-map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormaskDmap/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02/
-map/TensorArrayUnstack_1/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const?
!map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_2/element_shape?
map/TensorArrayV2_2TensorListReserve*map/TensorArrayV2_2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_2r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counter?
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_2:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *
bodyR
map_while_body_9223*
condR
map_while_cond_9222*!
output_shapes
: : : : : : : 2
	map/while?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   26
4map/TensorArrayV2Stack/TensorListStack/element_shape?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????a*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStackm
ShapeShape/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
ShapeR
Less/yConst*
_output_shapes
: *
dtype0*
value	B :#2
Less/yZ
LessLessShape:output:0Less/y:output:0*
T0*
_output_shapes
:2
Lesst
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceLess:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
condStatelessIfstrided_slice:output:0Shape:output:0/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :??????????????????a* 
_read_only_resource_inputs
 *"
else_branchR
cond_false_9333*3
output_shapes"
 :??????????????????a*!
then_branchR
cond_true_93322
condx
cond/IdentityIdentitycond:output:0*
T0*4
_output_shapes"
 :??????????????????a2
cond/Identityx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1e
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapecond/Identity:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????????????a:??????????????????:` \
4
_output_shapes"
 :??????????????????a
$
_user_specified_name
embeddings:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7872

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_8919
inputs_0
inputs_1

inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_80882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?Q
?
?__inference_model_layer_call_and_return_conditional_losses_8088

inputs
inputs_1

inputs_2
graph_convolution_8046
graph_convolution_1_8050
graph_convolution_2_8054
graph_convolution_3_8058
conv1d_8064
conv1d_8066
conv1d_1_8070
conv1d_1_8072

dense_8076

dense_8078
dense_1_8082
dense_1_8084
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?+graph_convolution_2/StatefulPartitionedCall?+graph_convolution_3/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_73752!
dropout/StatefulPartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0inputs_2graph_convolution_8046*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_74212+
)graph_convolution/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_74462#
!dropout_1/StatefulPartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0inputs_2graph_convolution_1_8050*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_74922-
+graph_convolution_1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_75172#
!dropout_2/StatefulPartitionedCall?
+graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0inputs_2graph_convolution_2_8054*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_75632-
+graph_convolution_2/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall4graph_convolution_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_75882#
!dropout_3/StatefulPartitionedCall?
+graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0inputs_2graph_convolution_3_8058*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_76342-
+graph_convolution_3/StatefulPartitionedCally
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat/concat/axis?
tf.concat/concatConcatV22graph_convolution/StatefulPartitionedCall:output:04graph_convolution_1/StatefulPartitionedCall:output:04graph_convolution_2/StatefulPartitionedCall:output:04graph_convolution_3/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
tf.concat/concat?
sort_pooling/PartitionedCallPartitionedCalltf.concat/concat:output:0inputs_1*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sort_pooling_layer_call_and_return_conditional_losses_78162
sort_pooling/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall%sort_pooling/PartitionedCall:output:0conv1d_8064conv1d_8066*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_78402 
conv1d/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_73512
max_pooling1d/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_8070conv1d_1_8072*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_78722"
 conv1d_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_78942
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_8076
dense_8078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_79132
dense/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_79412#
!dropout_4/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_1_8082dense_1_8084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_79702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall,^graph_convolution_2/StatefulPartitionedCall,^graph_convolution_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall2Z
+graph_convolution_2/StatefulPartitionedCall+graph_convolution_2/StatefulPartitionedCall2Z
+graph_convolution_3/StatefulPartitionedCall+graph_convolution_3/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
 sort_pooling_map_while_body_8699>
:sort_pooling_map_while_sort_pooling_map_while_loop_counter9
5sort_pooling_map_while_sort_pooling_map_strided_slice&
"sort_pooling_map_while_placeholder(
$sort_pooling_map_while_placeholder_1=
9sort_pooling_map_while_sort_pooling_map_strided_slice_1_0y
usort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0}
ysort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0#
sort_pooling_map_while_identity%
!sort_pooling_map_while_identity_1%
!sort_pooling_map_while_identity_2%
!sort_pooling_map_while_identity_3;
7sort_pooling_map_while_sort_pooling_map_strided_slice_1w
ssort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor{
wsort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor?
Hsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2J
Hsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shape?
:sort_pooling/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0"sort_pooling_map_while_placeholderQsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????a*
element_dtype02<
:sort_pooling/map/while/TensorArrayV2Read/TensorListGetItem?
Jsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2L
Jsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
<sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemysort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0"sort_pooling_map_while_placeholderSsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*#
_output_shapes
:?????????*
element_dtype0
2>
<sort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem?
)sort_pooling/map/while/boolean_mask/ShapeShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2+
)sort_pooling/map/while/boolean_mask/Shape?
7sort_pooling/map/while/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sort_pooling/map/while/boolean_mask/strided_slice/stack?
9sort_pooling/map/while/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sort_pooling/map/while/boolean_mask/strided_slice/stack_1?
9sort_pooling/map/while/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sort_pooling/map/while/boolean_mask/strided_slice/stack_2?
1sort_pooling/map/while/boolean_mask/strided_sliceStridedSlice2sort_pooling/map/while/boolean_mask/Shape:output:0@sort_pooling/map/while/boolean_mask/strided_slice/stack:output:0Bsort_pooling/map/while/boolean_mask/strided_slice/stack_1:output:0Bsort_pooling/map/while/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1sort_pooling/map/while/boolean_mask/strided_slice?
:sort_pooling/map/while/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sort_pooling/map/while/boolean_mask/Prod/reduction_indices?
(sort_pooling/map/while/boolean_mask/ProdProd:sort_pooling/map/while/boolean_mask/strided_slice:output:0Csort_pooling/map/while/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2*
(sort_pooling/map/while/boolean_mask/Prod?
+sort_pooling/map/while/boolean_mask/Shape_1ShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2-
+sort_pooling/map/while/boolean_mask/Shape_1?
9sort_pooling/map/while/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9sort_pooling/map/while/boolean_mask/strided_slice_1/stack?
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_1?
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sort_pooling/map/while/boolean_mask/strided_slice_1/stack_2?
3sort_pooling/map/while/boolean_mask/strided_slice_1StridedSlice4sort_pooling/map/while/boolean_mask/Shape_1:output:0Bsort_pooling/map/while/boolean_mask/strided_slice_1/stack:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_1/stack_1:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask25
3sort_pooling/map/while/boolean_mask/strided_slice_1?
+sort_pooling/map/while/boolean_mask/Shape_2ShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2-
+sort_pooling/map/while/boolean_mask/Shape_2?
9sort_pooling/map/while/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9sort_pooling/map/while/boolean_mask/strided_slice_2/stack?
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_1?
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sort_pooling/map/while/boolean_mask/strided_slice_2/stack_2?
3sort_pooling/map/while/boolean_mask/strided_slice_2StridedSlice4sort_pooling/map/while/boolean_mask/Shape_2:output:0Bsort_pooling/map/while/boolean_mask/strided_slice_2/stack:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_2/stack_1:output:0Dsort_pooling/map/while/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3sort_pooling/map/while/boolean_mask/strided_slice_2?
3sort_pooling/map/while/boolean_mask/concat/values_1Pack1sort_pooling/map/while/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:25
3sort_pooling/map/while/boolean_mask/concat/values_1?
/sort_pooling/map/while/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sort_pooling/map/while/boolean_mask/concat/axis?
*sort_pooling/map/while/boolean_mask/concatConcatV2<sort_pooling/map/while/boolean_mask/strided_slice_1:output:0<sort_pooling/map/while/boolean_mask/concat/values_1:output:0<sort_pooling/map/while/boolean_mask/strided_slice_2:output:08sort_pooling/map/while/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*sort_pooling/map/while/boolean_mask/concat?
+sort_pooling/map/while/boolean_mask/ReshapeReshapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:03sort_pooling/map/while/boolean_mask/concat:output:0*
T0*'
_output_shapes
:?????????a2-
+sort_pooling/map/while/boolean_mask/Reshape?
3sort_pooling/map/while/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3sort_pooling/map/while/boolean_mask/Reshape_1/shape?
-sort_pooling/map/while/boolean_mask/Reshape_1ReshapeCsort_pooling/map/while/TensorArrayV2Read_1/TensorListGetItem:item:0<sort_pooling/map/while/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2/
-sort_pooling/map/while/boolean_mask/Reshape_1?
)sort_pooling/map/while/boolean_mask/WhereWhere6sort_pooling/map/while/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2+
)sort_pooling/map/while/boolean_mask/Where?
+sort_pooling/map/while/boolean_mask/SqueezeSqueeze1sort_pooling/map/while/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2-
+sort_pooling/map/while/boolean_mask/Squeeze?
1sort_pooling/map/while/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sort_pooling/map/while/boolean_mask/GatherV2/axis?
,sort_pooling/map/while/boolean_mask/GatherV2GatherV24sort_pooling/map/while/boolean_mask/Reshape:output:04sort_pooling/map/while/boolean_mask/Squeeze:output:0:sort_pooling/map/while/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:?????????a2.
,sort_pooling/map/while/boolean_mask/GatherV2?
*sort_pooling/map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2,
*sort_pooling/map/while/strided_slice/stack?
,sort_pooling/map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,sort_pooling/map/while/strided_slice/stack_1?
,sort_pooling/map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,sort_pooling/map/while/strided_slice/stack_2?
$sort_pooling/map/while/strided_sliceStridedSlice5sort_pooling/map/while/boolean_mask/GatherV2:output:03sort_pooling/map/while/strided_slice/stack:output:05sort_pooling/map/while/strided_slice/stack_1:output:05sort_pooling/map/while/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2&
$sort_pooling/map/while/strided_slice?
#sort_pooling/map/while/argsort/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sort_pooling/map/while/argsort/axis?
$sort_pooling/map/while/argsort/ShapeShape-sort_pooling/map/while/strided_slice:output:0*
T0*
_output_shapes
:2&
$sort_pooling/map/while/argsort/Shape?
2sort_pooling/map/while/argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sort_pooling/map/while/argsort/strided_slice/stack?
4sort_pooling/map/while/argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sort_pooling/map/while/argsort/strided_slice/stack_1?
4sort_pooling/map/while/argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sort_pooling/map/while/argsort/strided_slice/stack_2?
,sort_pooling/map/while/argsort/strided_sliceStridedSlice-sort_pooling/map/while/argsort/Shape:output:0;sort_pooling/map/while/argsort/strided_slice/stack:output:0=sort_pooling/map/while/argsort/strided_slice/stack_1:output:0=sort_pooling/map/while/argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sort_pooling/map/while/argsort/strided_slice?
#sort_pooling/map/while/argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#sort_pooling/map/while/argsort/Rank?
%sort_pooling/map/while/argsort/TopKV2TopKV2-sort_pooling/map/while/strided_slice:output:05sort_pooling/map/while/argsort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????2'
%sort_pooling/map/while/argsort/TopKV2?
$sort_pooling/map/while/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$sort_pooling/map/while/GatherV2/axis?
sort_pooling/map/while/GatherV2GatherV2Asort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0/sort_pooling/map/while/argsort/TopKV2:indices:0-sort_pooling/map/while/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????a2!
sort_pooling/map/while/GatherV2?
sort_pooling/map/while/ShapeShapeAsort_pooling/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
sort_pooling/map/while/Shape?
,sort_pooling/map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sort_pooling/map/while/strided_slice_1/stack?
.sort_pooling/map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_1/stack_1?
.sort_pooling/map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_1/stack_2?
&sort_pooling/map/while/strided_slice_1StridedSlice%sort_pooling/map/while/Shape:output:05sort_pooling/map/while/strided_slice_1/stack:output:07sort_pooling/map/while/strided_slice_1/stack_1:output:07sort_pooling/map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sort_pooling/map/while/strided_slice_1?
sort_pooling/map/while/Shape_1Shape(sort_pooling/map/while/GatherV2:output:0*
T0*
_output_shapes
:2 
sort_pooling/map/while/Shape_1?
,sort_pooling/map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sort_pooling/map/while/strided_slice_2/stack?
.sort_pooling/map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_2/stack_1?
.sort_pooling/map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sort_pooling/map/while/strided_slice_2/stack_2?
&sort_pooling/map/while/strided_slice_2StridedSlice'sort_pooling/map/while/Shape_1:output:05sort_pooling/map/while/strided_slice_2/stack:output:07sort_pooling/map/while/strided_slice_2/stack_1:output:07sort_pooling/map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sort_pooling/map/while/strided_slice_2?
sort_pooling/map/while/subSub/sort_pooling/map/while/strided_slice_1:output:0/sort_pooling/map/while/strided_slice_2:output:0*
T0*
_output_shapes
: 2
sort_pooling/map/while/sub?
'sort_pooling/map/while/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sort_pooling/map/while/Pad/paddings/0/0?
%sort_pooling/map/while/Pad/paddings/0Pack0sort_pooling/map/while/Pad/paddings/0/0:output:0sort_pooling/map/while/sub:z:0*
N*
T0*
_output_shapes
:2'
%sort_pooling/map/while/Pad/paddings/0?
'sort_pooling/map/while/Pad/paddings/1_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'sort_pooling/map/while/Pad/paddings/1_1?
#sort_pooling/map/while/Pad/paddingsPack.sort_pooling/map/while/Pad/paddings/0:output:00sort_pooling/map/while/Pad/paddings/1_1:output:0*
N*
T0*
_output_shapes

:2%
#sort_pooling/map/while/Pad/paddings?
sort_pooling/map/while/PadPad(sort_pooling/map/while/GatherV2:output:0,sort_pooling/map/while/Pad/paddings:output:0*
T0*'
_output_shapes
:?????????a2
sort_pooling/map/while/Pad?
;sort_pooling/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sort_pooling_map_while_placeholder_1"sort_pooling_map_while_placeholder#sort_pooling/map/while/Pad:output:0*
_output_shapes
: *
element_dtype02=
;sort_pooling/map/while/TensorArrayV2Write/TensorListSetItem~
sort_pooling/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sort_pooling/map/while/add/y?
sort_pooling/map/while/addAddV2"sort_pooling_map_while_placeholder%sort_pooling/map/while/add/y:output:0*
T0*
_output_shapes
: 2
sort_pooling/map/while/add?
sort_pooling/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sort_pooling/map/while/add_1/y?
sort_pooling/map/while/add_1AddV2:sort_pooling_map_while_sort_pooling_map_while_loop_counter'sort_pooling/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sort_pooling/map/while/add_1?
sort_pooling/map/while/IdentityIdentity sort_pooling/map/while/add_1:z:0*
T0*
_output_shapes
: 2!
sort_pooling/map/while/Identity?
!sort_pooling/map/while/Identity_1Identity5sort_pooling_map_while_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2#
!sort_pooling/map/while/Identity_1?
!sort_pooling/map/while/Identity_2Identitysort_pooling/map/while/add:z:0*
T0*
_output_shapes
: 2#
!sort_pooling/map/while/Identity_2?
!sort_pooling/map/while/Identity_3IdentityKsort_pooling/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2#
!sort_pooling/map/while/Identity_3"K
sort_pooling_map_while_identity(sort_pooling/map/while/Identity:output:0"O
!sort_pooling_map_while_identity_1*sort_pooling/map/while/Identity_1:output:0"O
!sort_pooling_map_while_identity_2*sort_pooling/map/while/Identity_2:output:0"O
!sort_pooling_map_while_identity_3*sort_pooling/map/while/Identity_3:output:0"t
7sort_pooling_map_while_sort_pooling_map_strided_slice_19sort_pooling_map_while_sort_pooling_map_strided_slice_1_0"?
wsort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensorysort_pooling_map_while_tensorarrayv2read_1_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
ssort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensorusort_pooling_map_while_tensorarrayv2read_tensorlistgetitem_sort_pooling_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7446

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_9093

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :?????????????????? 2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
a
(__inference_dropout_4_layer_call_fn_9476

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_79412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
map_while_cond_7669$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_7669___redundant_placeholder0:
6map_while_map_while_cond_7669___redundant_placeholder1
map_while_identity
?
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less?
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
sort_pooling_cond_true_8808,
(sort_pooling_cond_sub_sort_pooling_shapeM
Isort_pooling_cond_pad_sort_pooling_map_tensorarrayv2stack_tensorliststack
sort_pooling_cond_identityt
sort_pooling/cond/sub/xConst*
_output_shapes
: *
dtype0*
value	B :#2
sort_pooling/cond/sub/x?
sort_pooling/cond/subSub sort_pooling/cond/sub/x:output:0(sort_pooling_cond_sub_sort_pooling_shape*
T0*
_output_shapes
:2
sort_pooling/cond/sub?
%sort_pooling/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%sort_pooling/cond/strided_slice/stack?
'sort_pooling/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sort_pooling/cond/strided_slice/stack_1?
'sort_pooling/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sort_pooling/cond/strided_slice/stack_2?
sort_pooling/cond/strided_sliceStridedSlicesort_pooling/cond/sub:z:0.sort_pooling/cond/strided_slice/stack:output:00sort_pooling/cond/strided_slice/stack_1:output:00sort_pooling/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sort_pooling/cond/strided_slice?
"sort_pooling/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sort_pooling/cond/Pad/paddings/1/0?
 sort_pooling/cond/Pad/paddings/1Pack+sort_pooling/cond/Pad/paddings/1/0:output:0(sort_pooling/cond/strided_slice:output:0*
N*
T0*
_output_shapes
:2"
 sort_pooling/cond/Pad/paddings/1?
"sort_pooling/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"sort_pooling/cond/Pad/paddings/0_1?
"sort_pooling/cond/Pad/paddings/2_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"sort_pooling/cond/Pad/paddings/2_1?
sort_pooling/cond/Pad/paddingsPack+sort_pooling/cond/Pad/paddings/0_1:output:0)sort_pooling/cond/Pad/paddings/1:output:0+sort_pooling/cond/Pad/paddings/2_1:output:0*
N*
T0*
_output_shapes

:2 
sort_pooling/cond/Pad/paddings?
sort_pooling/cond/PadPadIsort_pooling_cond_pad_sort_pooling_map_tensorarrayv2stack_tensorliststack'sort_pooling/cond/Pad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Pad?
sort_pooling/cond/IdentityIdentitysort_pooling/cond/Pad:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Identity"A
sort_pooling_cond_identity#sort_pooling/cond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?
b
C__inference_dropout_4_layer_call_and_return_conditional_losses_9466

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
 sort_pooling_map_while_cond_8698>
:sort_pooling_map_while_sort_pooling_map_while_loop_counter9
5sort_pooling_map_while_sort_pooling_map_strided_slice&
"sort_pooling_map_while_placeholder(
$sort_pooling_map_while_placeholder_1>
:sort_pooling_map_while_less_sort_pooling_map_strided_sliceT
Psort_pooling_map_while_sort_pooling_map_while_cond_8698___redundant_placeholder0T
Psort_pooling_map_while_sort_pooling_map_while_cond_8698___redundant_placeholder1#
sort_pooling_map_while_identity
?
sort_pooling/map/while/LessLess"sort_pooling_map_while_placeholder:sort_pooling_map_while_less_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2
sort_pooling/map/while/Less?
sort_pooling/map/while/Less_1Less:sort_pooling_map_while_sort_pooling_map_while_loop_counter5sort_pooling_map_while_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2
sort_pooling/map/while/Less_1?
!sort_pooling/map/while/LogicalAnd
LogicalAnd!sort_pooling/map/while/Less_1:z:0sort_pooling/map/while/Less:z:0*
_output_shapes
: 2#
!sort_pooling/map/while/LogicalAnd?
sort_pooling/map/while/IdentityIdentity%sort_pooling/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2!
sort_pooling/map/while/Identity"K
sort_pooling_map_while_identity(sort_pooling/map/while/Identity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
b
C__inference_dropout_4_layer_call_and_return_conditional_losses_7941

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
sort_pooling_cond_false_8493!
sort_pooling_cond_placeholderW
Ssort_pooling_cond_strided_slice_sort_pooling_map_tensorarrayv2stack_tensorliststack
sort_pooling_cond_identity?
%sort_pooling/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2'
%sort_pooling/cond/strided_slice/stack?
'sort_pooling/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2)
'sort_pooling/cond/strided_slice/stack_1?
'sort_pooling/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2)
'sort_pooling/cond/strided_slice/stack_2?
sort_pooling/cond/strided_sliceStridedSliceSsort_pooling_cond_strided_slice_sort_pooling_map_tensorarrayv2stack_tensorliststack.sort_pooling/cond/strided_slice/stack:output:00sort_pooling/cond/strided_slice/stack_1:output:00sort_pooling/cond/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????a*

begin_mask*
end_mask2!
sort_pooling/cond/strided_slice?
sort_pooling/cond/IdentityIdentity(sort_pooling/cond/strided_slice:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Identity"A
sort_pooling_cond_identity#sort_pooling/cond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_7588

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_7380

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_7563

inputs
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpr
MatMulBatchMatMulV2inputs_1inputs*
T0*4
_output_shapes"
 :?????????????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_8967

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
z
%__inference_conv1d_layer_call_fn_9399

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_78402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_graph_convolution_1_layer_call_fn_9076
inputs_0
inputs_1
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_74922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
B
&__inference_flatten_layer_call_fn_9434

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_78942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_9445

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
y
$__inference_dense_layer_call_fn_9454

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_79132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?^
?
__inference__traced_save_9661
file_prefix7
3savev2_graph_convolution_kernel_read_readvariableop9
5savev2_graph_convolution_1_kernel_read_readvariableop9
5savev2_graph_convolution_2_kernel_read_readvariableop9
5savev2_graph_convolution_3_kernel_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_adam_graph_convolution_kernel_m_read_readvariableop@
<savev2_adam_graph_convolution_1_kernel_m_read_readvariableop@
<savev2_adam_graph_convolution_2_kernel_m_read_readvariableop@
<savev2_adam_graph_convolution_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop>
:savev2_adam_graph_convolution_kernel_v_read_readvariableop@
<savev2_adam_graph_convolution_1_kernel_v_read_readvariableop@
<savev2_adam_graph_convolution_2_kernel_v_read_readvariableop@
<savev2_adam_graph_convolution_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_graph_convolution_kernel_read_readvariableop5savev2_graph_convolution_1_kernel_read_readvariableop5savev2_graph_convolution_2_kernel_read_readvariableop5savev2_graph_convolution_3_kernel_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_graph_convolution_kernel_m_read_readvariableop<savev2_adam_graph_convolution_1_kernel_m_read_readvariableop<savev2_adam_graph_convolution_2_kernel_m_read_readvariableop<savev2_adam_graph_convolution_3_kernel_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop:savev2_adam_graph_convolution_kernel_v_read_readvariableop<savev2_adam_graph_convolution_1_kernel_v_read_readvariableop<savev2_adam_graph_convolution_2_kernel_v_read_readvariableop<savev2_adam_graph_convolution_3_kernel_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : :  :  : :a:: : :
??:?:	?:: : : : : : : : : : :  :  : :a:: : :
??:?:	?:: :  :  : :a:: : :
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: :$ 

_output_shapes

:  :$ 

_output_shapes

:  :$ 

_output_shapes

: :($
"
_output_shapes
:a: 

_output_shapes
::($
"
_output_shapes
: : 

_output_shapes
: :&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

:  :$ 

_output_shapes

:  :$ 

_output_shapes

: :($
"
_output_shapes
:a: 

_output_shapes
::($
"
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::$" 

_output_shapes

: :$# 

_output_shapes

:  :$$ 

_output_shapes

:  :$% 

_output_shapes

: :(&$
"
_output_shapes
:a: '

_output_shapes
::(($
"
_output_shapes
: : )

_output_shapes
: :&*"
 
_output_shapes
:
??:!+

_output_shapes	
:?:%,!

_output_shapes
:	?: -

_output_shapes
::.

_output_shapes
: 
?
?
sort_pooling_cond_true_8492,
(sort_pooling_cond_sub_sort_pooling_shapeM
Isort_pooling_cond_pad_sort_pooling_map_tensorarrayv2stack_tensorliststack
sort_pooling_cond_identityt
sort_pooling/cond/sub/xConst*
_output_shapes
: *
dtype0*
value	B :#2
sort_pooling/cond/sub/x?
sort_pooling/cond/subSub sort_pooling/cond/sub/x:output:0(sort_pooling_cond_sub_sort_pooling_shape*
T0*
_output_shapes
:2
sort_pooling/cond/sub?
%sort_pooling/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%sort_pooling/cond/strided_slice/stack?
'sort_pooling/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sort_pooling/cond/strided_slice/stack_1?
'sort_pooling/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sort_pooling/cond/strided_slice/stack_2?
sort_pooling/cond/strided_sliceStridedSlicesort_pooling/cond/sub:z:0.sort_pooling/cond/strided_slice/stack:output:00sort_pooling/cond/strided_slice/stack_1:output:00sort_pooling/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sort_pooling/cond/strided_slice?
"sort_pooling/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sort_pooling/cond/Pad/paddings/1/0?
 sort_pooling/cond/Pad/paddings/1Pack+sort_pooling/cond/Pad/paddings/1/0:output:0(sort_pooling/cond/strided_slice:output:0*
N*
T0*
_output_shapes
:2"
 sort_pooling/cond/Pad/paddings/1?
"sort_pooling/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"sort_pooling/cond/Pad/paddings/0_1?
"sort_pooling/cond/Pad/paddings/2_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"sort_pooling/cond/Pad/paddings/2_1?
sort_pooling/cond/Pad/paddingsPack+sort_pooling/cond/Pad/paddings/0_1:output:0)sort_pooling/cond/Pad/paddings/1:output:0+sort_pooling/cond/Pad/paddings/2_1:output:0*
N*
T0*
_output_shapes

:2 
sort_pooling/cond/Pad/paddings?
sort_pooling/cond/PadPadIsort_pooling_cond_pad_sort_pooling_map_tensorarrayv2stack_tensorliststack'sort_pooling/cond/Pad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Pad?
sort_pooling/cond/IdentityIdentitysort_pooling/cond/Pad:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Identity"A
sort_pooling_cond_identity#sort_pooling/cond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?
D
(__inference_dropout_4_layer_call_fn_9481

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_79462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
 sort_pooling_map_while_cond_8382>
:sort_pooling_map_while_sort_pooling_map_while_loop_counter9
5sort_pooling_map_while_sort_pooling_map_strided_slice&
"sort_pooling_map_while_placeholder(
$sort_pooling_map_while_placeholder_1>
:sort_pooling_map_while_less_sort_pooling_map_strided_sliceT
Psort_pooling_map_while_sort_pooling_map_while_cond_8382___redundant_placeholder0T
Psort_pooling_map_while_sort_pooling_map_while_cond_8382___redundant_placeholder1#
sort_pooling_map_while_identity
?
sort_pooling/map/while/LessLess"sort_pooling_map_while_placeholder:sort_pooling_map_while_less_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2
sort_pooling/map/while/Less?
sort_pooling/map/while/Less_1Less:sort_pooling_map_while_sort_pooling_map_while_loop_counter5sort_pooling_map_while_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2
sort_pooling/map/while/Less_1?
!sort_pooling/map/while/LogicalAnd
LogicalAnd!sort_pooling/map/while/Less_1:z:0sort_pooling/map/while/Less:z:0*
_output_shapes
: 2#
!sort_pooling/map/while/LogicalAnd?
sort_pooling/map/while/IdentityIdentity%sort_pooling/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2!
sort_pooling/map/while/Identity"K
sort_pooling_map_while_identity(sort_pooling/map/while/Identity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?I
?
?__inference_model_layer_call_and_return_conditional_losses_8035
input_1
input_2

input_3
graph_convolution_7993
graph_convolution_1_7997
graph_convolution_2_8001
graph_convolution_3_8005
conv1d_8011
conv1d_8013
conv1d_1_8017
conv1d_1_8019

dense_8023

dense_8025
dense_1_8029
dense_1_8031
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?+graph_convolution_2/StatefulPartitionedCall?+graph_convolution_3/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_73802
dropout/PartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0input_3graph_convolution_7993*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_74212+
)graph_convolution/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_74512
dropout_1/PartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0input_3graph_convolution_1_7997*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_74922-
+graph_convolution_1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_75222
dropout_2/PartitionedCall?
+graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0input_3graph_convolution_2_8001*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_75632-
+graph_convolution_2/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall4graph_convolution_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_75932
dropout_3/PartitionedCall?
+graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0input_3graph_convolution_3_8005*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_76342-
+graph_convolution_3/StatefulPartitionedCally
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat/concat/axis?
tf.concat/concatConcatV22graph_convolution/StatefulPartitionedCall:output:04graph_convolution_1/StatefulPartitionedCall:output:04graph_convolution_2/StatefulPartitionedCall:output:04graph_convolution_3/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
tf.concat/concat?
sort_pooling/PartitionedCallPartitionedCalltf.concat/concat:output:0input_2*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sort_pooling_layer_call_and_return_conditional_losses_78162
sort_pooling/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall%sort_pooling/PartitionedCall:output:0conv1d_8011conv1d_8013*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_78402 
conv1d/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_73512
max_pooling1d/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_8017conv1d_1_8019*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_78722"
 conv1d_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_78942
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_8023
dense_8025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_79132
dense/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_79462
dropout_4/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_1_8029dense_1_8031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_79702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall,^graph_convolution_2/StatefulPartitionedCall,^graph_convolution_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall2Z
+graph_convolution_2/StatefulPartitionedCall+graph_convolution_2/StatefulPartitionedCall2Z
+graph_convolution_3/StatefulPartitionedCall+graph_convolution_3/StatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_7840

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:a*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:a2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
a2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_9005
inputs_0
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpt
MatMulBatchMatMulV2inputs_1inputs_0*
T0*4
_output_shapes"
 :??????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:??????????????????:'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
0__inference_graph_convolution_layer_call_fn_9013
inputs_0
inputs_1
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_74212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:??????????????????:'???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
c
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_7351

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_9131
inputs_0
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpt
MatMulBatchMatMulV2inputs_1inputs_0*
T0*4
_output_shapes"
 :?????????????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
!model_sort_pooling_cond_true_72628
4model_sort_pooling_cond_sub_model_sort_pooling_shapeY
Umodel_sort_pooling_cond_pad_model_sort_pooling_map_tensorarrayv2stack_tensorliststack$
 model_sort_pooling_cond_identity?
model/sort_pooling/cond/sub/xConst*
_output_shapes
: *
dtype0*
value	B :#2
model/sort_pooling/cond/sub/x?
model/sort_pooling/cond/subSub&model/sort_pooling/cond/sub/x:output:04model_sort_pooling_cond_sub_model_sort_pooling_shape*
T0*
_output_shapes
:2
model/sort_pooling/cond/sub?
+model/sort_pooling/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model/sort_pooling/cond/strided_slice/stack?
-model/sort_pooling/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/sort_pooling/cond/strided_slice/stack_1?
-model/sort_pooling/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/sort_pooling/cond/strided_slice/stack_2?
%model/sort_pooling/cond/strided_sliceStridedSlicemodel/sort_pooling/cond/sub:z:04model/sort_pooling/cond/strided_slice/stack:output:06model/sort_pooling/cond/strided_slice/stack_1:output:06model/sort_pooling/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/sort_pooling/cond/strided_slice?
(model/sort_pooling/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2*
(model/sort_pooling/cond/Pad/paddings/1/0?
&model/sort_pooling/cond/Pad/paddings/1Pack1model/sort_pooling/cond/Pad/paddings/1/0:output:0.model/sort_pooling/cond/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&model/sort_pooling/cond/Pad/paddings/1?
(model/sort_pooling/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(model/sort_pooling/cond/Pad/paddings/0_1?
(model/sort_pooling/cond/Pad/paddings/2_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(model/sort_pooling/cond/Pad/paddings/2_1?
$model/sort_pooling/cond/Pad/paddingsPack1model/sort_pooling/cond/Pad/paddings/0_1:output:0/model/sort_pooling/cond/Pad/paddings/1:output:01model/sort_pooling/cond/Pad/paddings/2_1:output:0*
N*
T0*
_output_shapes

:2&
$model/sort_pooling/cond/Pad/paddings?
model/sort_pooling/cond/PadPadUmodel_sort_pooling_cond_pad_model_sort_pooling_map_tensorarrayv2stack_tensorliststack-model/sort_pooling/cond/Pad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????a2
model/sort_pooling/cond/Pad?
 model/sort_pooling/cond/IdentityIdentity$model/sort_pooling/cond/Pad:output:0*
T0*4
_output_shapes"
 :??????????????????a2"
 model/sort_pooling/cond/Identity"M
 model_sort_pooling_cond_identity)model/sort_pooling/cond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?
l
cond_true_7779
cond_sub_shape3
/cond_pad_map_tensorarrayv2stack_tensorliststack
cond_identityZ

cond/sub/xConst*
_output_shapes
: *
dtype0*
value	B :#2

cond/sub/xe
cond/subSubcond/sub/x:output:0cond_sub_shape*
T0*
_output_shapes
:2

cond/sub~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack?
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1?
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2?
cond/strided_sliceStridedSlicecond/sub:z:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicep
cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Pad/paddings/1/0?
cond/Pad/paddings/1Packcond/Pad/paddings/1/0:output:0cond/strided_slice:output:0*
N*
T0*
_output_shapes
:2
cond/Pad/paddings/1
cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2
cond/Pad/paddings/0_1
cond/Pad/paddings/2_1Const*
_output_shapes
:*
dtype0*
valueB"        2
cond/Pad/paddings/2_1?
cond/Pad/paddingsPackcond/Pad/paddings/0_1:output:0cond/Pad/paddings/1:output:0cond/Pad/paddings/2_1:output:0*
N*
T0*
_output_shapes

:2
cond/Pad/paddings?
cond/PadPad/cond_pad_map_tensorarrayv2stack_tensorliststackcond/Pad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????a2

cond/Pad|
cond/IdentityIdentitycond/Pad:output:0*
T0*4
_output_shapes"
 :??????????????????a2
cond/Identity"'
cond_identitycond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
ڐ
?

__inference__wrapped_model_7342
input_1
input_2

input_3;
7model_graph_convolution_shape_2_readvariableop_resource=
9model_graph_convolution_1_shape_2_readvariableop_resource=
9model_graph_convolution_2_shape_2_readvariableop_resource=
9model_graph_convolution_3_shape_2_readvariableop_resource<
8model_conv1d_conv1d_expanddims_1_readvariableop_resource0
,model_conv1d_biasadd_readvariableop_resource>
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_1_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource
identity??#model/conv1d/BiasAdd/ReadVariableOp?/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?%model/conv1d_1/BiasAdd/ReadVariableOp?1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?0model/graph_convolution/transpose/ReadVariableOp?2model/graph_convolution_1/transpose/ReadVariableOp?2model/graph_convolution_2/transpose/ReadVariableOp?2model/graph_convolution_3/transpose/ReadVariableOp?
model/dropout/IdentityIdentityinput_1*
T0*4
_output_shapes"
 :??????????????????2
model/dropout/Identity?
model/graph_convolution/MatMulBatchMatMulV2input_3model/dropout/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2 
model/graph_convolution/MatMul?
model/graph_convolution/ShapeShape'model/graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
model/graph_convolution/Shape?
model/graph_convolution/Shape_1Shape'model/graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2!
model/graph_convolution/Shape_1?
model/graph_convolution/unstackUnpack(model/graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2!
model/graph_convolution/unstack?
.model/graph_convolution/Shape_2/ReadVariableOpReadVariableOp7model_graph_convolution_shape_2_readvariableop_resource*
_output_shapes

: *
dtype020
.model/graph_convolution/Shape_2/ReadVariableOp?
model/graph_convolution/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2!
model/graph_convolution/Shape_2?
!model/graph_convolution/unstack_1Unpack(model/graph_convolution/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2#
!model/graph_convolution/unstack_1?
%model/graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2'
%model/graph_convolution/Reshape/shape?
model/graph_convolution/ReshapeReshape'model/graph_convolution/MatMul:output:0.model/graph_convolution/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2!
model/graph_convolution/Reshape?
0model/graph_convolution/transpose/ReadVariableOpReadVariableOp7model_graph_convolution_shape_2_readvariableop_resource*
_output_shapes

: *
dtype022
0model/graph_convolution/transpose/ReadVariableOp?
&model/graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model/graph_convolution/transpose/perm?
!model/graph_convolution/transpose	Transpose8model/graph_convolution/transpose/ReadVariableOp:value:0/model/graph_convolution/transpose/perm:output:0*
T0*
_output_shapes

: 2#
!model/graph_convolution/transpose?
'model/graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2)
'model/graph_convolution/Reshape_1/shape?
!model/graph_convolution/Reshape_1Reshape%model/graph_convolution/transpose:y:00model/graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2#
!model/graph_convolution/Reshape_1?
 model/graph_convolution/MatMul_1MatMul(model/graph_convolution/Reshape:output:0*model/graph_convolution/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2"
 model/graph_convolution/MatMul_1?
)model/graph_convolution/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2+
)model/graph_convolution/Reshape_2/shape/2?
'model/graph_convolution/Reshape_2/shapePack(model/graph_convolution/unstack:output:0(model/graph_convolution/unstack:output:12model/graph_convolution/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2)
'model/graph_convolution/Reshape_2/shape?
!model/graph_convolution/Reshape_2Reshape*model/graph_convolution/MatMul_1:product:00model/graph_convolution/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2#
!model/graph_convolution/Reshape_2?
model/graph_convolution/TanhTanh*model/graph_convolution/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
model/graph_convolution/Tanh?
model/dropout_1/IdentityIdentity model/graph_convolution/Tanh:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
model/dropout_1/Identity?
 model/graph_convolution_1/MatMulBatchMatMulV2input_3!model/dropout_1/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2"
 model/graph_convolution_1/MatMul?
model/graph_convolution_1/ShapeShape)model/graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2!
model/graph_convolution_1/Shape?
!model/graph_convolution_1/Shape_1Shape)model/graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2#
!model/graph_convolution_1/Shape_1?
!model/graph_convolution_1/unstackUnpack*model/graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2#
!model/graph_convolution_1/unstack?
0model/graph_convolution_1/Shape_2/ReadVariableOpReadVariableOp9model_graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype022
0model/graph_convolution_1/Shape_2/ReadVariableOp?
!model/graph_convolution_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2#
!model/graph_convolution_1/Shape_2?
#model/graph_convolution_1/unstack_1Unpack*model/graph_convolution_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2%
#model/graph_convolution_1/unstack_1?
'model/graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2)
'model/graph_convolution_1/Reshape/shape?
!model/graph_convolution_1/ReshapeReshape)model/graph_convolution_1/MatMul:output:00model/graph_convolution_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2#
!model/graph_convolution_1/Reshape?
2model/graph_convolution_1/transpose/ReadVariableOpReadVariableOp9model_graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype024
2model/graph_convolution_1/transpose/ReadVariableOp?
(model/graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(model/graph_convolution_1/transpose/perm?
#model/graph_convolution_1/transpose	Transpose:model/graph_convolution_1/transpose/ReadVariableOp:value:01model/graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:  2%
#model/graph_convolution_1/transpose?
)model/graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2+
)model/graph_convolution_1/Reshape_1/shape?
#model/graph_convolution_1/Reshape_1Reshape'model/graph_convolution_1/transpose:y:02model/graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:  2%
#model/graph_convolution_1/Reshape_1?
"model/graph_convolution_1/MatMul_1MatMul*model/graph_convolution_1/Reshape:output:0,model/graph_convolution_1/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2$
"model/graph_convolution_1/MatMul_1?
+model/graph_convolution_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2-
+model/graph_convolution_1/Reshape_2/shape/2?
)model/graph_convolution_1/Reshape_2/shapePack*model/graph_convolution_1/unstack:output:0*model/graph_convolution_1/unstack:output:14model/graph_convolution_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)model/graph_convolution_1/Reshape_2/shape?
#model/graph_convolution_1/Reshape_2Reshape,model/graph_convolution_1/MatMul_1:product:02model/graph_convolution_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2%
#model/graph_convolution_1/Reshape_2?
model/graph_convolution_1/TanhTanh,model/graph_convolution_1/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
model/graph_convolution_1/Tanh?
model/dropout_2/IdentityIdentity"model/graph_convolution_1/Tanh:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
model/dropout_2/Identity?
 model/graph_convolution_2/MatMulBatchMatMulV2input_3!model/dropout_2/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2"
 model/graph_convolution_2/MatMul?
model/graph_convolution_2/ShapeShape)model/graph_convolution_2/MatMul:output:0*
T0*
_output_shapes
:2!
model/graph_convolution_2/Shape?
!model/graph_convolution_2/Shape_1Shape)model/graph_convolution_2/MatMul:output:0*
T0*
_output_shapes
:2#
!model/graph_convolution_2/Shape_1?
!model/graph_convolution_2/unstackUnpack*model/graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2#
!model/graph_convolution_2/unstack?
0model/graph_convolution_2/Shape_2/ReadVariableOpReadVariableOp9model_graph_convolution_2_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype022
0model/graph_convolution_2/Shape_2/ReadVariableOp?
!model/graph_convolution_2/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2#
!model/graph_convolution_2/Shape_2?
#model/graph_convolution_2/unstack_1Unpack*model/graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2%
#model/graph_convolution_2/unstack_1?
'model/graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2)
'model/graph_convolution_2/Reshape/shape?
!model/graph_convolution_2/ReshapeReshape)model/graph_convolution_2/MatMul:output:00model/graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2#
!model/graph_convolution_2/Reshape?
2model/graph_convolution_2/transpose/ReadVariableOpReadVariableOp9model_graph_convolution_2_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype024
2model/graph_convolution_2/transpose/ReadVariableOp?
(model/graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(model/graph_convolution_2/transpose/perm?
#model/graph_convolution_2/transpose	Transpose:model/graph_convolution_2/transpose/ReadVariableOp:value:01model/graph_convolution_2/transpose/perm:output:0*
T0*
_output_shapes

:  2%
#model/graph_convolution_2/transpose?
)model/graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2+
)model/graph_convolution_2/Reshape_1/shape?
#model/graph_convolution_2/Reshape_1Reshape'model/graph_convolution_2/transpose:y:02model/graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:  2%
#model/graph_convolution_2/Reshape_1?
"model/graph_convolution_2/MatMul_1MatMul*model/graph_convolution_2/Reshape:output:0,model/graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2$
"model/graph_convolution_2/MatMul_1?
+model/graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2-
+model/graph_convolution_2/Reshape_2/shape/2?
)model/graph_convolution_2/Reshape_2/shapePack*model/graph_convolution_2/unstack:output:0*model/graph_convolution_2/unstack:output:14model/graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)model/graph_convolution_2/Reshape_2/shape?
#model/graph_convolution_2/Reshape_2Reshape,model/graph_convolution_2/MatMul_1:product:02model/graph_convolution_2/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2%
#model/graph_convolution_2/Reshape_2?
model/graph_convolution_2/TanhTanh,model/graph_convolution_2/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
model/graph_convolution_2/Tanh?
model/dropout_3/IdentityIdentity"model/graph_convolution_2/Tanh:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
model/dropout_3/Identity?
 model/graph_convolution_3/MatMulBatchMatMulV2input_3!model/dropout_3/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2"
 model/graph_convolution_3/MatMul?
model/graph_convolution_3/ShapeShape)model/graph_convolution_3/MatMul:output:0*
T0*
_output_shapes
:2!
model/graph_convolution_3/Shape?
!model/graph_convolution_3/Shape_1Shape)model/graph_convolution_3/MatMul:output:0*
T0*
_output_shapes
:2#
!model/graph_convolution_3/Shape_1?
!model/graph_convolution_3/unstackUnpack*model/graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2#
!model/graph_convolution_3/unstack?
0model/graph_convolution_3/Shape_2/ReadVariableOpReadVariableOp9model_graph_convolution_3_shape_2_readvariableop_resource*
_output_shapes

: *
dtype022
0model/graph_convolution_3/Shape_2/ReadVariableOp?
!model/graph_convolution_3/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/graph_convolution_3/Shape_2?
#model/graph_convolution_3/unstack_1Unpack*model/graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2%
#model/graph_convolution_3/unstack_1?
'model/graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2)
'model/graph_convolution_3/Reshape/shape?
!model/graph_convolution_3/ReshapeReshape)model/graph_convolution_3/MatMul:output:00model/graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2#
!model/graph_convolution_3/Reshape?
2model/graph_convolution_3/transpose/ReadVariableOpReadVariableOp9model_graph_convolution_3_shape_2_readvariableop_resource*
_output_shapes

: *
dtype024
2model/graph_convolution_3/transpose/ReadVariableOp?
(model/graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(model/graph_convolution_3/transpose/perm?
#model/graph_convolution_3/transpose	Transpose:model/graph_convolution_3/transpose/ReadVariableOp:value:01model/graph_convolution_3/transpose/perm:output:0*
T0*
_output_shapes

: 2%
#model/graph_convolution_3/transpose?
)model/graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2+
)model/graph_convolution_3/Reshape_1/shape?
#model/graph_convolution_3/Reshape_1Reshape'model/graph_convolution_3/transpose:y:02model/graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2%
#model/graph_convolution_3/Reshape_1?
"model/graph_convolution_3/MatMul_1MatMul*model/graph_convolution_3/Reshape:output:0,model/graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2$
"model/graph_convolution_3/MatMul_1?
+model/graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+model/graph_convolution_3/Reshape_2/shape/2?
)model/graph_convolution_3/Reshape_2/shapePack*model/graph_convolution_3/unstack:output:0*model/graph_convolution_3/unstack:output:14model/graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)model/graph_convolution_3/Reshape_2/shape?
#model/graph_convolution_3/Reshape_2Reshape,model/graph_convolution_3/MatMul_1:product:02model/graph_convolution_3/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2%
#model/graph_convolution_3/Reshape_2?
model/graph_convolution_3/TanhTanh,model/graph_convolution_3/Reshape_2:output:0*
T0*4
_output_shapes"
 :??????????????????2 
model/graph_convolution_3/Tanh?
model/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
model/tf.concat/concat/axis?
model/tf.concat/concatConcatV2 model/graph_convolution/Tanh:y:0"model/graph_convolution_1/Tanh:y:0"model/graph_convolution_2/Tanh:y:0"model/graph_convolution_3/Tanh:y:0$model/tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
model/tf.concat/concat?
model/sort_pooling/map/ShapeShapemodel/tf.concat/concat:output:0*
T0*
_output_shapes
:2
model/sort_pooling/map/Shape?
*model/sort_pooling/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/sort_pooling/map/strided_slice/stack?
,model/sort_pooling/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/sort_pooling/map/strided_slice/stack_1?
,model/sort_pooling/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/sort_pooling/map/strided_slice/stack_2?
$model/sort_pooling/map/strided_sliceStridedSlice%model/sort_pooling/map/Shape:output:03model/sort_pooling/map/strided_slice/stack:output:05model/sort_pooling/map/strided_slice/stack_1:output:05model/sort_pooling/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/sort_pooling/map/strided_slice?
2model/sort_pooling/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2model/sort_pooling/map/TensorArrayV2/element_shape?
$model/sort_pooling/map/TensorArrayV2TensorListReserve;model/sort_pooling/map/TensorArrayV2/element_shape:output:0-model/sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$model/sort_pooling/map/TensorArrayV2?
4model/sort_pooling/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4model/sort_pooling/map/TensorArrayV2_1/element_shape?
&model/sort_pooling/map/TensorArrayV2_1TensorListReserve=model/sort_pooling/map/TensorArrayV2_1/element_shape:output:0-model/sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02(
&model/sort_pooling/map/TensorArrayV2_1?
Lmodel/sort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2N
Lmodel/sort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
>model/sort_pooling/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/tf.concat/concat:output:0Umodel/sort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>model/sort_pooling/map/TensorArrayUnstack/TensorListFromTensor?
Nmodel/sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2P
Nmodel/sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
@model/sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinput_2Wmodel/sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02B
@model/sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor~
model/sort_pooling/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
model/sort_pooling/map/Const?
4model/sort_pooling/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4model/sort_pooling/map/TensorArrayV2_2/element_shape?
&model/sort_pooling/map/TensorArrayV2_2TensorListReserve=model/sort_pooling/map/TensorArrayV2_2/element_shape:output:0-model/sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02(
&model/sort_pooling/map/TensorArrayV2_2?
)model/sort_pooling/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model/sort_pooling/map/while/loop_counter?
model/sort_pooling/map/whileStatelessWhile2model/sort_pooling/map/while/loop_counter:output:0-model/sort_pooling/map/strided_slice:output:0%model/sort_pooling/map/Const:output:0/model/sort_pooling/map/TensorArrayV2_2:handle:0-model/sort_pooling/map/strided_slice:output:0Nmodel/sort_pooling/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Pmodel/sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *2
body*R(
&model_sort_pooling_map_while_body_7153*2
cond*R(
&model_sort_pooling_map_while_cond_7152*!
output_shapes
: : : : : : : 2
model/sort_pooling/map/while?
Gmodel/sort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2I
Gmodel/sort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shape?
9model/sort_pooling/map/TensorArrayV2Stack/TensorListStackTensorListStack%model/sort_pooling/map/while:output:3Pmodel/sort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????a*
element_dtype02;
9model/sort_pooling/map/TensorArrayV2Stack/TensorListStack?
model/sort_pooling/ShapeShapeBmodel/sort_pooling/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
model/sort_pooling/Shapex
model/sort_pooling/Less/yConst*
_output_shapes
: *
dtype0*
value	B :#2
model/sort_pooling/Less/y?
model/sort_pooling/LessLess!model/sort_pooling/Shape:output:0"model/sort_pooling/Less/y:output:0*
T0*
_output_shapes
:2
model/sort_pooling/Less?
&model/sort_pooling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&model/sort_pooling/strided_slice/stack?
(model/sort_pooling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/sort_pooling/strided_slice/stack_1?
(model/sort_pooling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/sort_pooling/strided_slice/stack_2?
 model/sort_pooling/strided_sliceStridedSlicemodel/sort_pooling/Less:z:0/model/sort_pooling/strided_slice/stack:output:01model/sort_pooling/strided_slice/stack_1:output:01model/sort_pooling/strided_slice/stack_2:output:0*
Index0*
T0
*
_output_shapes
: *
shrink_axis_mask2"
 model/sort_pooling/strided_slice?
model/sort_pooling/condStatelessIf)model/sort_pooling/strided_slice:output:0!model/sort_pooling/Shape:output:0Bmodel/sort_pooling/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :??????????????????a* 
_read_only_resource_inputs
 *5
else_branch&R$
"model_sort_pooling_cond_false_7263*3
output_shapes"
 :??????????????????a*4
then_branch%R#
!model_sort_pooling_cond_true_72622
model/sort_pooling/cond?
 model/sort_pooling/cond/IdentityIdentity model/sort_pooling/cond:output:0*
T0*4
_output_shapes"
 :??????????????????a2"
 model/sort_pooling/cond/Identity?
(model/sort_pooling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/sort_pooling/strided_slice_1/stack?
*model/sort_pooling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/sort_pooling/strided_slice_1/stack_1?
*model/sort_pooling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/sort_pooling/strided_slice_1/stack_2?
"model/sort_pooling/strided_slice_1StridedSlice!model/sort_pooling/Shape:output:01model/sort_pooling/strided_slice_1/stack:output:03model/sort_pooling/strided_slice_1/stack_1:output:03model/sort_pooling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model/sort_pooling/strided_slice_1?
"model/sort_pooling/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model/sort_pooling/Reshape/shape/1?
"model/sort_pooling/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model/sort_pooling/Reshape/shape/2?
 model/sort_pooling/Reshape/shapePack+model/sort_pooling/strided_slice_1:output:0+model/sort_pooling/Reshape/shape/1:output:0+model/sort_pooling/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model/sort_pooling/Reshape/shape?
model/sort_pooling/ReshapeReshape)model/sort_pooling/cond/Identity:output:0)model/sort_pooling/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
model/sort_pooling/Reshape?
"model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model/conv1d/conv1d/ExpandDims/dim?
model/conv1d/conv1d/ExpandDims
ExpandDims#model/sort_pooling/Reshape:output:0+model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2 
model/conv1d/conv1d/ExpandDims?
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:a*
dtype021
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
$model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv1d/conv1d/ExpandDims_1/dim?
 model/conv1d/conv1d/ExpandDims_1
ExpandDims7model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:a2"
 model/conv1d/conv1d/ExpandDims_1?
model/conv1d/conv1dConv2D'model/conv1d/conv1d/ExpandDims:output:0)model/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
a2
model/conv1d/conv1d?
model/conv1d/conv1d/SqueezeSqueezemodel/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
model/conv1d/conv1d/Squeeze?
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv1d/BiasAdd/ReadVariableOp?
model/conv1d/BiasAddBiasAdd$model/conv1d/conv1d/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2
model/conv1d/BiasAdd?
"model/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/max_pooling1d/ExpandDims/dim?
model/max_pooling1d/ExpandDims
ExpandDimsmodel/conv1d/BiasAdd:output:0+model/max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2 
model/max_pooling1d/ExpandDims?
model/max_pooling1d/MaxPoolMaxPool'model/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/max_pooling1d/MaxPool?
model/max_pooling1d/SqueezeSqueeze$model/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2
model/max_pooling1d/Squeeze?
$model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/conv1d_1/conv1d/ExpandDims/dim?
 model/conv1d_1/conv1d/ExpandDims
ExpandDims$model/max_pooling1d/Squeeze:output:0-model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2"
 model/conv1d_1/conv1d/ExpandDims?
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype023
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_1/conv1d/ExpandDims_1/dim?
"model/conv1d_1/conv1d/ExpandDims_1
ExpandDims9model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2$
"model/conv1d_1/conv1d/ExpandDims_1?
model/conv1d_1/conv1dConv2D)model/conv1d_1/conv1d/ExpandDims:output:0+model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
model/conv1d_1/conv1d?
model/conv1d_1/conv1d/SqueezeSqueezemodel/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
model/conv1d_1/conv1d/Squeeze?
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv1d_1/BiasAdd/ReadVariableOp?
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/conv1d/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
model/conv1d_1/BiasAdd{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten/Const?
model/flatten/ReshapeReshapemodel/conv1d_1/BiasAdd:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshape?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense/BiasAdd}
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense/Relu?
model/dropout_4/IdentityIdentitymodel/dense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_4/Identity?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/BiasAdd?
model/dense_1/SigmoidSigmoidmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_1/Sigmoid?
IdentityIdentitymodel/dense_1/Sigmoid:y:0$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/conv1d/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp1^model/graph_convolution/transpose/ReadVariableOp3^model/graph_convolution_1/transpose/ReadVariableOp3^model/graph_convolution_2/transpose/ReadVariableOp3^model/graph_convolution_3/transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2d
0model/graph_convolution/transpose/ReadVariableOp0model/graph_convolution/transpose/ReadVariableOp2h
2model/graph_convolution_1/transpose/ReadVariableOp2model/graph_convolution_1/transpose/ReadVariableOp2h
2model/graph_convolution_2/transpose/ReadVariableOp2model/graph_convolution_2/transpose/ReadVariableOp2h
2model/graph_convolution_3/transpose/ReadVariableOp2model/graph_convolution_3/transpose/ReadVariableOp:] Y
4
_output_shapes"
 :??????????????????
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_7517

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_7593

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :?????????????????? 2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_9068
inputs_0
inputs_1#
shape_2_readvariableop_resource
identity??transpose/ReadVariableOpt
MatMulBatchMatMulV2inputs_1inputs_0*
T0*4
_output_shapes"
 :?????????????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:  *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2g
TanhTanhReshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Tanh?
IdentityIdentityTanh:y:0^transpose/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:?????????????????? :'???????????????????????????:24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_7913

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_3_layer_call_fn_9161

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_75882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_8235
input_1
input_2

input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_73422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_9030

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :?????????????????? 2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :?????????????????? :\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
&model_sort_pooling_map_while_cond_7152J
Fmodel_sort_pooling_map_while_model_sort_pooling_map_while_loop_counterE
Amodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice,
(model_sort_pooling_map_while_placeholder.
*model_sort_pooling_map_while_placeholder_1J
Fmodel_sort_pooling_map_while_less_model_sort_pooling_map_strided_slice`
\model_sort_pooling_map_while_model_sort_pooling_map_while_cond_7152___redundant_placeholder0`
\model_sort_pooling_map_while_model_sort_pooling_map_while_cond_7152___redundant_placeholder1)
%model_sort_pooling_map_while_identity
?
!model/sort_pooling/map/while/LessLess(model_sort_pooling_map_while_placeholderFmodel_sort_pooling_map_while_less_model_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2#
!model/sort_pooling/map/while/Less?
#model/sort_pooling/map/while/Less_1LessFmodel_sort_pooling_map_while_model_sort_pooling_map_while_loop_counterAmodel_sort_pooling_map_while_model_sort_pooling_map_strided_slice*
T0*
_output_shapes
: 2%
#model/sort_pooling/map/while/Less_1?
'model/sort_pooling/map/while/LogicalAnd
LogicalAnd'model/sort_pooling/map/while/Less_1:z:0%model/sort_pooling/map/while/Less:z:0*
_output_shapes
: 2)
'model/sort_pooling/map/while/LogicalAnd?
%model/sort_pooling/map/while/IdentityIdentity+model/sort_pooling/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2'
%model/sort_pooling/map/while/Identity"W
%model_sort_pooling_map_while_identity.model/sort_pooling/map/while/Identity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
??
?	
?__inference_model_layer_call_and_return_conditional_losses_8579
inputs_0
inputs_1

inputs_25
1graph_convolution_shape_2_readvariableop_resource7
3graph_convolution_1_shape_2_readvariableop_resource7
3graph_convolution_2_shape_2_readvariableop_resource7
3graph_convolution_3_shape_2_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?*graph_convolution/transpose/ReadVariableOp?,graph_convolution_1/transpose/ReadVariableOp?,graph_convolution_2/transpose/ReadVariableOp?,graph_convolution_3/transpose/ReadVariableOps
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulinputs_0dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/dropout/Mulf
dropout/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/dropout/Mul_1?
graph_convolution/MatMulBatchMatMulV2inputs_2dropout/dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_convolution/MatMul?
graph_convolution/ShapeShape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape?
graph_convolution/Shape_1Shape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape_1?
graph_convolution/unstackUnpack"graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution/unstack?
(graph_convolution/Shape_2/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02*
(graph_convolution/Shape_2/ReadVariableOp?
graph_convolution/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_convolution/Shape_2?
graph_convolution/unstack_1Unpack"graph_convolution/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution/unstack_1?
graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2!
graph_convolution/Reshape/shape?
graph_convolution/ReshapeReshape!graph_convolution/MatMul:output:0(graph_convolution/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
graph_convolution/Reshape?
*graph_convolution/transpose/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02,
*graph_convolution/transpose/ReadVariableOp?
 graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2"
 graph_convolution/transpose/perm?
graph_convolution/transpose	Transpose2graph_convolution/transpose/ReadVariableOp:value:0)graph_convolution/transpose/perm:output:0*
T0*
_output_shapes

: 2
graph_convolution/transpose?
!graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2#
!graph_convolution/Reshape_1/shape?
graph_convolution/Reshape_1Reshapegraph_convolution/transpose:y:0*graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
graph_convolution/Reshape_1?
graph_convolution/MatMul_1MatMul"graph_convolution/Reshape:output:0$graph_convolution/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution/MatMul_1?
#graph_convolution/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#graph_convolution/Reshape_2/shape/2?
!graph_convolution/Reshape_2/shapePack"graph_convolution/unstack:output:0"graph_convolution/unstack:output:1,graph_convolution/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!graph_convolution/Reshape_2/shape?
graph_convolution/Reshape_2Reshape$graph_convolution/MatMul_1:product:0*graph_convolution/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution/Reshape_2?
graph_convolution/TanhTanh$graph_convolution/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution/Tanhw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulgraph_convolution/Tanh:y:0 dropout_1/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapegraph_convolution/Tanh:y:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_1/dropout/Mul_1?
graph_convolution_1/MatMulBatchMatMulV2inputs_2dropout_1/dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_1/MatMul?
graph_convolution_1/ShapeShape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape?
graph_convolution_1/Shape_1Shape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape_1?
graph_convolution_1/unstackUnpack$graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_1/unstack?
*graph_convolution_1/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02,
*graph_convolution_1/Shape_2/ReadVariableOp?
graph_convolution_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2
graph_convolution_1/Shape_2?
graph_convolution_1/unstack_1Unpack$graph_convolution_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_1/unstack_1?
!graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!graph_convolution_1/Reshape/shape?
graph_convolution_1/ReshapeReshape#graph_convolution_1/MatMul:output:0*graph_convolution_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_1/Reshape?
,graph_convolution_1/transpose/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02.
,graph_convolution_1/transpose/ReadVariableOp?
"graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_1/transpose/perm?
graph_convolution_1/transpose	Transpose4graph_convolution_1/transpose/ReadVariableOp:value:0+graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
graph_convolution_1/transpose?
#graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2%
#graph_convolution_1/Reshape_1/shape?
graph_convolution_1/Reshape_1Reshape!graph_convolution_1/transpose:y:0,graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
graph_convolution_1/Reshape_1?
graph_convolution_1/MatMul_1MatMul$graph_convolution_1/Reshape:output:0&graph_convolution_1/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_1/MatMul_1?
%graph_convolution_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2'
%graph_convolution_1/Reshape_2/shape/2?
#graph_convolution_1/Reshape_2/shapePack$graph_convolution_1/unstack:output:0$graph_convolution_1/unstack:output:1.graph_convolution_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_1/Reshape_2/shape?
graph_convolution_1/Reshape_2Reshape&graph_convolution_1/MatMul_1:product:0,graph_convolution_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_1/Reshape_2?
graph_convolution_1/TanhTanh&graph_convolution_1/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_1/Tanhw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulgraph_convolution_1/Tanh:y:0 dropout_2/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_2/dropout/Mul~
dropout_2/dropout/ShapeShapegraph_convolution_1/Tanh:y:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_2/dropout/Mul_1?
graph_convolution_2/MatMulBatchMatMulV2inputs_2dropout_2/dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_2/MatMul?
graph_convolution_2/ShapeShape#graph_convolution_2/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_2/Shape?
graph_convolution_2/Shape_1Shape#graph_convolution_2/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_2/Shape_1?
graph_convolution_2/unstackUnpack$graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_2/unstack?
*graph_convolution_2/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_2_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02,
*graph_convolution_2/Shape_2/ReadVariableOp?
graph_convolution_2/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"        2
graph_convolution_2/Shape_2?
graph_convolution_2/unstack_1Unpack$graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_2/unstack_1?
!graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!graph_convolution_2/Reshape/shape?
graph_convolution_2/ReshapeReshape#graph_convolution_2/MatMul:output:0*graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_2/Reshape?
,graph_convolution_2/transpose/ReadVariableOpReadVariableOp3graph_convolution_2_shape_2_readvariableop_resource*
_output_shapes

:  *
dtype02.
,graph_convolution_2/transpose/ReadVariableOp?
"graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_2/transpose/perm?
graph_convolution_2/transpose	Transpose4graph_convolution_2/transpose/ReadVariableOp:value:0+graph_convolution_2/transpose/perm:output:0*
T0*
_output_shapes

:  2
graph_convolution_2/transpose?
#graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2%
#graph_convolution_2/Reshape_1/shape?
graph_convolution_2/Reshape_1Reshape!graph_convolution_2/transpose:y:0,graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:  2
graph_convolution_2/Reshape_1?
graph_convolution_2/MatMul_1MatMul$graph_convolution_2/Reshape:output:0&graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_2/MatMul_1?
%graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2'
%graph_convolution_2/Reshape_2/shape/2?
#graph_convolution_2/Reshape_2/shapePack$graph_convolution_2/unstack:output:0$graph_convolution_2/unstack:output:1.graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_2/Reshape_2/shape?
graph_convolution_2/Reshape_2Reshape&graph_convolution_2/MatMul_1:product:0,graph_convolution_2/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_2/Reshape_2?
graph_convolution_2/TanhTanh&graph_convolution_2/Reshape_2:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_2/Tanhw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMulgraph_convolution_2/Tanh:y:0 dropout_3/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_3/dropout/Mul~
dropout_3/dropout/ShapeShapegraph_convolution_2/Tanh:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_3/dropout/Mul_1?
graph_convolution_3/MatMulBatchMatMulV2inputs_2dropout_3/dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_convolution_3/MatMul?
graph_convolution_3/ShapeShape#graph_convolution_3/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_3/Shape?
graph_convolution_3/Shape_1Shape#graph_convolution_3/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_3/Shape_1?
graph_convolution_3/unstackUnpack$graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_3/unstack?
*graph_convolution_3/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_3_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02,
*graph_convolution_3/Shape_2/ReadVariableOp?
graph_convolution_3/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_convolution_3/Shape_2?
graph_convolution_3/unstack_1Unpack$graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_3/unstack_1?
!graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!graph_convolution_3/Reshape/shape?
graph_convolution_3/ReshapeReshape#graph_convolution_3/MatMul:output:0*graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_convolution_3/Reshape?
,graph_convolution_3/transpose/ReadVariableOpReadVariableOp3graph_convolution_3_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02.
,graph_convolution_3/transpose/ReadVariableOp?
"graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_3/transpose/perm?
graph_convolution_3/transpose	Transpose4graph_convolution_3/transpose/ReadVariableOp:value:0+graph_convolution_3/transpose/perm:output:0*
T0*
_output_shapes

: 2
graph_convolution_3/transpose?
#graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2%
#graph_convolution_3/Reshape_1/shape?
graph_convolution_3/Reshape_1Reshape!graph_convolution_3/transpose:y:0,graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
graph_convolution_3/Reshape_1?
graph_convolution_3/MatMul_1MatMul$graph_convolution_3/Reshape:output:0&graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
graph_convolution_3/MatMul_1?
%graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%graph_convolution_3/Reshape_2/shape/2?
#graph_convolution_3/Reshape_2/shapePack$graph_convolution_3/unstack:output:0$graph_convolution_3/unstack:output:1.graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_3/Reshape_2/shape?
graph_convolution_3/Reshape_2Reshape&graph_convolution_3/MatMul_1:product:0,graph_convolution_3/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_convolution_3/Reshape_2?
graph_convolution_3/TanhTanh&graph_convolution_3/Reshape_2:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_convolution_3/Tanhy
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat/concat/axis?
tf.concat/concatConcatV2graph_convolution/Tanh:y:0graph_convolution_1/Tanh:y:0graph_convolution_2/Tanh:y:0graph_convolution_3/Tanh:y:0tf.concat/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????????????a2
tf.concat/concaty
sort_pooling/map/ShapeShapetf.concat/concat:output:0*
T0*
_output_shapes
:2
sort_pooling/map/Shape?
$sort_pooling/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sort_pooling/map/strided_slice/stack?
&sort_pooling/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sort_pooling/map/strided_slice/stack_1?
&sort_pooling/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sort_pooling/map/strided_slice/stack_2?
sort_pooling/map/strided_sliceStridedSlicesort_pooling/map/Shape:output:0-sort_pooling/map/strided_slice/stack:output:0/sort_pooling/map/strided_slice/stack_1:output:0/sort_pooling/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sort_pooling/map/strided_slice?
,sort_pooling/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sort_pooling/map/TensorArrayV2/element_shape?
sort_pooling/map/TensorArrayV2TensorListReserve5sort_pooling/map/TensorArrayV2/element_shape:output:0'sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sort_pooling/map/TensorArrayV2?
.sort_pooling/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sort_pooling/map/TensorArrayV2_1/element_shape?
 sort_pooling/map/TensorArrayV2_1TensorListReserve7sort_pooling/map/TensorArrayV2_1/element_shape:output:0'sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02"
 sort_pooling/map/TensorArrayV2_1?
Fsort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2H
Fsort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
8sort_pooling/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensortf.concat/concat:output:0Osort_pooling/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8sort_pooling/map/TensorArrayUnstack/TensorListFromTensor?
Hsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2J
Hsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
:sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_1Qsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02<
:sort_pooling/map/TensorArrayUnstack_1/TensorListFromTensorr
sort_pooling/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
sort_pooling/map/Const?
.sort_pooling/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sort_pooling/map/TensorArrayV2_2/element_shape?
 sort_pooling/map/TensorArrayV2_2TensorListReserve7sort_pooling/map/TensorArrayV2_2/element_shape:output:0'sort_pooling/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sort_pooling/map/TensorArrayV2_2?
#sort_pooling/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sort_pooling/map/while/loop_counter?
sort_pooling/map/whileStatelessWhile,sort_pooling/map/while/loop_counter:output:0'sort_pooling/map/strided_slice:output:0sort_pooling/map/Const:output:0)sort_pooling/map/TensorArrayV2_2:handle:0'sort_pooling/map/strided_slice:output:0Hsort_pooling/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsort_pooling/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *,
body$R"
 sort_pooling_map_while_body_8383*,
cond$R"
 sort_pooling_map_while_cond_8382*!
output_shapes
: : : : : : : 2
sort_pooling/map/while?
Asort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2C
Asort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shape?
3sort_pooling/map/TensorArrayV2Stack/TensorListStackTensorListStacksort_pooling/map/while:output:3Jsort_pooling/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????a*
element_dtype025
3sort_pooling/map/TensorArrayV2Stack/TensorListStack?
sort_pooling/ShapeShape<sort_pooling/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
sort_pooling/Shapel
sort_pooling/Less/yConst*
_output_shapes
: *
dtype0*
value	B :#2
sort_pooling/Less/y?
sort_pooling/LessLesssort_pooling/Shape:output:0sort_pooling/Less/y:output:0*
T0*
_output_shapes
:2
sort_pooling/Less?
 sort_pooling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sort_pooling/strided_slice/stack?
"sort_pooling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sort_pooling/strided_slice/stack_1?
"sort_pooling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sort_pooling/strided_slice/stack_2?
sort_pooling/strided_sliceStridedSlicesort_pooling/Less:z:0)sort_pooling/strided_slice/stack:output:0+sort_pooling/strided_slice/stack_1:output:0+sort_pooling/strided_slice/stack_2:output:0*
Index0*
T0
*
_output_shapes
: *
shrink_axis_mask2
sort_pooling/strided_slice?
sort_pooling/condStatelessIf#sort_pooling/strided_slice:output:0sort_pooling/Shape:output:0<sort_pooling/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :??????????????????a* 
_read_only_resource_inputs
 */
else_branch R
sort_pooling_cond_false_8493*3
output_shapes"
 :??????????????????a*.
then_branchR
sort_pooling_cond_true_84922
sort_pooling/cond?
sort_pooling/cond/IdentityIdentitysort_pooling/cond:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Identity?
"sort_pooling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sort_pooling/strided_slice_1/stack?
$sort_pooling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sort_pooling/strided_slice_1/stack_1?
$sort_pooling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sort_pooling/strided_slice_1/stack_2?
sort_pooling/strided_slice_1StridedSlicesort_pooling/Shape:output:0+sort_pooling/strided_slice_1/stack:output:0-sort_pooling/strided_slice_1/stack_1:output:0-sort_pooling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sort_pooling/strided_slice_1
sort_pooling/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
sort_pooling/Reshape/shape/1~
sort_pooling/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
sort_pooling/Reshape/shape/2?
sort_pooling/Reshape/shapePack%sort_pooling/strided_slice_1:output:0%sort_pooling/Reshape/shape/1:output:0%sort_pooling/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
sort_pooling/Reshape/shape?
sort_pooling/ReshapeReshape#sort_pooling/cond/Identity:output:0#sort_pooling/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
sort_pooling/Reshape?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimssort_pooling/Reshape:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:a*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:a2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
a2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2
conv1d/BiasAdd~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d/BiasAdd:output:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2
max_pooling1d/Squeeze?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsmax_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d_1/BiasAddo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv1d_1/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/dropout/Const?
dropout_4/dropout/MulMuldense/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/dropout/Mulz
dropout_4/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp+^graph_convolution/transpose/ReadVariableOp-^graph_convolution_1/transpose/ReadVariableOp-^graph_convolution_2/transpose/ReadVariableOp-^graph_convolution_3/transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????????????:??????????????????:'???????????????????????????::::::::::::2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2X
*graph_convolution/transpose/ReadVariableOp*graph_convolution/transpose/ReadVariableOp2\
,graph_convolution_1/transpose/ReadVariableOp,graph_convolution_1/transpose/ReadVariableOp2\
,graph_convolution_2/transpose/ReadVariableOp,graph_convolution_2/transpose/ReadVariableOp2\
,graph_convolution_3/transpose/ReadVariableOp,graph_convolution_3/transpose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?
a
(__inference_dropout_1_layer_call_fn_9035

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_74462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?r
?
map_while_body_9223$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensora
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????a   2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????a*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItem?
=map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
=map/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*#
_output_shapes
:?????????*
element_dtype0
21
/map/while/TensorArrayV2Read_1/TensorListGetItem?
map/while/boolean_mask/ShapeShape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
map/while/boolean_mask/Shape?
*map/while/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*map/while/boolean_mask/strided_slice/stack?
,map/while/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,map/while/boolean_mask/strided_slice/stack_1?
,map/while/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,map/while/boolean_mask/strided_slice/stack_2?
$map/while/boolean_mask/strided_sliceStridedSlice%map/while/boolean_mask/Shape:output:03map/while/boolean_mask/strided_slice/stack:output:05map/while/boolean_mask/strided_slice/stack_1:output:05map/while/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2&
$map/while/boolean_mask/strided_slice?
-map/while/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2/
-map/while/boolean_mask/Prod/reduction_indices?
map/while/boolean_mask/ProdProd-map/while/boolean_mask/strided_slice:output:06map/while/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
map/while/boolean_mask/Prod?
map/while/boolean_mask/Shape_1Shape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
map/while/boolean_mask/Shape_1?
,map/while/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,map/while/boolean_mask/strided_slice_1/stack?
.map/while/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.map/while/boolean_mask/strided_slice_1/stack_1?
.map/while/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.map/while/boolean_mask/strided_slice_1/stack_2?
&map/while/boolean_mask/strided_slice_1StridedSlice'map/while/boolean_mask/Shape_1:output:05map/while/boolean_mask/strided_slice_1/stack:output:07map/while/boolean_mask/strided_slice_1/stack_1:output:07map/while/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2(
&map/while/boolean_mask/strided_slice_1?
map/while/boolean_mask/Shape_2Shape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
map/while/boolean_mask/Shape_2?
,map/while/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,map/while/boolean_mask/strided_slice_2/stack?
.map/while/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.map/while/boolean_mask/strided_slice_2/stack_1?
.map/while/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.map/while/boolean_mask/strided_slice_2/stack_2?
&map/while/boolean_mask/strided_slice_2StridedSlice'map/while/boolean_mask/Shape_2:output:05map/while/boolean_mask/strided_slice_2/stack:output:07map/while/boolean_mask/strided_slice_2/stack_1:output:07map/while/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&map/while/boolean_mask/strided_slice_2?
&map/while/boolean_mask/concat/values_1Pack$map/while/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2(
&map/while/boolean_mask/concat/values_1?
"map/while/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"map/while/boolean_mask/concat/axis?
map/while/boolean_mask/concatConcatV2/map/while/boolean_mask/strided_slice_1:output:0/map/while/boolean_mask/concat/values_1:output:0/map/while/boolean_mask/strided_slice_2:output:0+map/while/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
map/while/boolean_mask/concat?
map/while/boolean_mask/ReshapeReshape4map/while/TensorArrayV2Read/TensorListGetItem:item:0&map/while/boolean_mask/concat:output:0*
T0*'
_output_shapes
:?????????a2 
map/while/boolean_mask/Reshape?
&map/while/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&map/while/boolean_mask/Reshape_1/shape?
 map/while/boolean_mask/Reshape_1Reshape6map/while/TensorArrayV2Read_1/TensorListGetItem:item:0/map/while/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2"
 map/while/boolean_mask/Reshape_1?
map/while/boolean_mask/WhereWhere)map/while/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2
map/while/boolean_mask/Where?
map/while/boolean_mask/SqueezeSqueeze$map/while/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2 
map/while/boolean_mask/Squeeze?
$map/while/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$map/while/boolean_mask/GatherV2/axis?
map/while/boolean_mask/GatherV2GatherV2'map/while/boolean_mask/Reshape:output:0'map/while/boolean_mask/Squeeze:output:0-map/while/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:?????????a2!
map/while/boolean_mask/GatherV2?
map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
map/while/strided_slice/stack?
map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
map/while/strided_slice/stack_1?
map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
map/while/strided_slice/stack_2?
map/while/strided_sliceStridedSlice(map/while/boolean_mask/GatherV2:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
map/while/strided_slicer
map/while/argsort/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/argsort/axis?
map/while/argsort/ShapeShape map/while/strided_slice:output:0*
T0*
_output_shapes
:2
map/while/argsort/Shape?
%map/while/argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%map/while/argsort/strided_slice/stack?
'map/while/argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'map/while/argsort/strided_slice/stack_1?
'map/while/argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'map/while/argsort/strided_slice/stack_2?
map/while/argsort/strided_sliceStridedSlice map/while/argsort/Shape:output:0.map/while/argsort/strided_slice/stack:output:00map/while/argsort/strided_slice/stack_1:output:00map/while/argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
map/while/argsort/strided_slicer
map/while/argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/argsort/Rank?
map/while/argsort/TopKV2TopKV2 map/while/strided_slice:output:0(map/while/argsort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????2
map/while/argsort/TopKV2t
map/while/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/GatherV2/axis?
map/while/GatherV2GatherV24map/while/TensorArrayV2Read/TensorListGetItem:item:0"map/while/argsort/TopKV2:indices:0 map/while/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????a2
map/while/GatherV2?
map/while/ShapeShape4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
map/while/Shape?
map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
map/while/strided_slice_1/stack?
!map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_1/stack_1?
!map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_1/stack_2?
map/while/strided_slice_1StridedSlicemap/while/Shape:output:0(map/while/strided_slice_1/stack:output:0*map/while/strided_slice_1/stack_1:output:0*map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/while/strided_slice_1q
map/while/Shape_1Shapemap/while/GatherV2:output:0*
T0*
_output_shapes
:2
map/while/Shape_1?
map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
map/while/strided_slice_2/stack?
!map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_2/stack_1?
!map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!map/while/strided_slice_2/stack_2?
map/while/strided_slice_2StridedSlicemap/while/Shape_1:output:0(map/while/strided_slice_2/stack:output:0*map/while/strided_slice_2/stack_1:output:0*map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/while/strided_slice_2?
map/while/subSub"map/while/strided_slice_1:output:0"map/while/strided_slice_2:output:0*
T0*
_output_shapes
: 2
map/while/subz
map/while/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
map/while/Pad/paddings/0/0?
map/while/Pad/paddings/0Pack#map/while/Pad/paddings/0/0:output:0map/while/sub:z:0*
N*
T0*
_output_shapes
:2
map/while/Pad/paddings/0?
map/while/Pad/paddings/1_1Const*
_output_shapes
:*
dtype0*
valueB"        2
map/while/Pad/paddings/1_1?
map/while/Pad/paddingsPack!map/while/Pad/paddings/0:output:0#map/while/Pad/paddings/1_1:output:0*
N*
T0*
_output_shapes

:2
map/while/Pad/paddings?
map/while/PadPadmap/while/GatherV2:output:0map/while/Pad/paddings:output:0*
T0*'
_output_shapes
:?????????a2
map/while/Pad?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/Pad:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/y?
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1j
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: 2
map/while/Identityv
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Identity_1l
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 2
map/while/Identity_2?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
map/while/Identity_3"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"?
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
a
(__inference_dropout_2_layer_call_fn_9098

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_75172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????????????? 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_9429

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
sort_pooling_cond_false_8809!
sort_pooling_cond_placeholderW
Ssort_pooling_cond_strided_slice_sort_pooling_map_tensorarrayv2stack_tensorliststack
sort_pooling_cond_identity?
%sort_pooling/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2'
%sort_pooling/cond/strided_slice/stack?
'sort_pooling/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2)
'sort_pooling/cond/strided_slice/stack_1?
'sort_pooling/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2)
'sort_pooling/cond/strided_slice/stack_2?
sort_pooling/cond/strided_sliceStridedSliceSsort_pooling_cond_strided_slice_sort_pooling_map_tensorarrayv2stack_tensorliststack.sort_pooling/cond/strided_slice/stack:output:00sort_pooling/cond/strided_slice/stack_1:output:00sort_pooling/cond/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????a*

begin_mask*
end_mask2!
sort_pooling/cond/strided_slice?
sort_pooling/cond/IdentityIdentity(sort_pooling/cond/strided_slice:output:0*
T0*4
_output_shapes"
 :??????????????????a2
sort_pooling/cond/Identity"A
sort_pooling_cond_identity#sort_pooling/cond/Identity:output:0*9
_input_shapes(
&::??????????????????a:  

_output_shapes
:::6
4
_output_shapes"
 :??????????????????a
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_9492

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_7946

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
input_1=
serving_default_input_1:0??????????????????
D
input_29
serving_default_input_2:0
??????????????????
Q
input_3F
serving_default_input_3:0'???????????????????????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?{
_tf_keras_network?{{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution", "inbound_nodes": [[["dropout", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["graph_convolution", 0, 0, {}]]]}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution_1", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution_1", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["graph_convolution_1", 0, 0, {}]]]}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution_2", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution_2", "inbound_nodes": [[["dropout_2", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["graph_convolution_2", 0, 0, {}]]]}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution_3", "trainable": true, "dtype": "float32", "units": 1, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution_3", "inbound_nodes": [[["dropout_3", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["graph_convolution", 0, 0, {"axis": -1}], ["graph_convolution_1", 0, 0, {"axis": -1}], ["graph_convolution_2", 0, 0, {"axis": -1}], ["graph_convolution_3", 0, 0, {"axis": -1}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "bool", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "SortPooling", "config": {"k": 35, "flatten_output": true}, "name": "sort_pooling", "inbound_nodes": [[["tf.concat", 0, 0, {"mask": ["input_2", 0, 0]}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [97]}, "strides": {"class_name": "__tuple__", "items": [97]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["sort_pooling", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 1]}, {"class_name": "TensorShape", "items": [null, null]}, {"class_name": "TensorShape", "items": [null, null, null]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution", "inbound_nodes": [[["dropout", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["graph_convolution", 0, 0, {}]]]}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution_1", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution_1", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["graph_convolution_1", 0, 0, {}]]]}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution_2", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution_2", "inbound_nodes": [[["dropout_2", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["graph_convolution_2", 0, 0, {}]]]}, {"class_name": "GraphConvolution", "config": {"name": "graph_convolution_3", "trainable": true, "dtype": "float32", "units": 1, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "graph_convolution_3", "inbound_nodes": [[["dropout_3", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["graph_convolution", 0, 0, {"axis": -1}], ["graph_convolution_1", 0, 0, {"axis": -1}], ["graph_convolution_2", 0, 0, {"axis": -1}], ["graph_convolution_3", 0, 0, {"axis": -1}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "bool", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "SortPooling", "config": {"k": 35, "flatten_output": true}, "name": "sort_pooling", "inbound_nodes": [[["tf.concat", 0, 0, {"mask": ["input_2", 0, 0]}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [97]}, "strides": {"class_name": "__tuple__", "items": [97]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["sort_pooling", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?

kernel
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GraphConvolution", "name": "graph_convolution", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "graph_convolution", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 1]}, {"class_name": "TensorShape", "items": [null, null, null]}]}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?

(kernel
)	variables
*regularization_losses
+trainable_variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GraphConvolution", "name": "graph_convolution_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "graph_convolution_1", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 32]}, {"class_name": "TensorShape", "items": [null, null, null]}]}
?
-	variables
.regularization_losses
/trainable_variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?

1kernel
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GraphConvolution", "name": "graph_convolution_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "graph_convolution_2", "trainable": true, "dtype": "float32", "units": 32, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 32]}, {"class_name": "TensorShape", "items": [null, null, null]}]}
?
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GraphConvolution", "name": "graph_convolution_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "graph_convolution_3", "trainable": true, "dtype": "float32", "units": 1, "use_bias": false, "activation": "tanh", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 32]}, {"class_name": "TensorShape", "items": [null, null, null]}]}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "bool", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "bool", "sparse": false, "ragged": false, "name": "input_2"}}
?
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SortPooling", "name": "sort_pooling", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"k": 35, "flatten_output": true}}
?	

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [97]}, "strides": {"class_name": "__tuple__", "items": [97]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3395, 1]}}
?
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Nkernel
Obias
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17, 16]}}
?
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Xkernel
Ybias
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 416}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 416]}}
?
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratem?(m?1m?:m?Dm?Em?Nm?Om?Xm?Ym?bm?cm?v?(v?1v?:v?Dv?Ev?Nv?Ov?Xv?Yv?bv?cv?"
	optimizer
v
0
(1
12
:3
D4
E5
N6
O7
X8
Y9
b10
c11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
(1
12
:3
D4
E5
N6
O7
X8
Y9
b10
c11"
trackable_list_wrapper
?
	variables
regularization_losses
mnon_trainable_variables
trainable_variables
nlayer_regularization_losses

olayers
player_metrics
qmetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
rnon_trainable_variables
trainable_variables
slayer_regularization_losses

tlayers
ulayer_metrics
vmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2graph_convolution/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
 	variables
!regularization_losses
wnon_trainable_variables
"trainable_variables
xlayer_regularization_losses

ylayers
zlayer_metrics
{metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$	variables
%regularization_losses
|non_trainable_variables
&trainable_variables
}layer_regularization_losses

~layers
layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2graph_convolution_1/kernel
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
?
)	variables
*regularization_losses
?non_trainable_variables
+trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-	variables
.regularization_losses
?non_trainable_variables
/trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2graph_convolution_2/kernel
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
?
2	variables
3regularization_losses
?non_trainable_variables
4trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6	variables
7regularization_losses
?non_trainable_variables
8trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* 2graph_convolution_3/kernel
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
?
;	variables
<regularization_losses
?non_trainable_variables
=trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@	variables
Aregularization_losses
?non_trainable_variables
Btrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!a2conv1d/kernel
:2conv1d/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
F	variables
Gregularization_losses
?non_trainable_variables
Htrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
J	variables
Kregularization_losses
?non_trainable_variables
Ltrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:# 2conv1d_1/kernel
: 2conv1d_1/bias
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
P	variables
Qregularization_losses
?non_trainable_variables
Rtrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
T	variables
Uregularization_losses
?non_trainable_variables
Vtrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2dense/kernel
:?2
dense/bias
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
Z	variables
[regularization_losses
?non_trainable_variables
\trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^	variables
_regularization_losses
?non_trainable_variables
`trainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_1/kernel
:2dense_1/bias
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
?
d	variables
eregularization_losses
?non_trainable_variables
ftrainable_variables
 ?layer_regularization_losses
?layers
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/graph_convolution/kernel/m
1:/  2!Adam/graph_convolution_1/kernel/m
1:/  2!Adam/graph_convolution_2/kernel/m
1:/ 2!Adam/graph_convolution_3/kernel/m
(:&a2Adam/conv1d/kernel/m
:2Adam/conv1d/bias/m
*:( 2Adam/conv1d_1/kernel/m
 : 2Adam/conv1d_1/bias/m
%:#
??2Adam/dense/kernel/m
:?2Adam/dense/bias/m
&:$	?2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
/:- 2Adam/graph_convolution/kernel/v
1:/  2!Adam/graph_convolution_1/kernel/v
1:/  2!Adam/graph_convolution_2/kernel/v
1:/ 2!Adam/graph_convolution_3/kernel/v
(:&a2Adam/conv1d/kernel/v
:2Adam/conv1d/bias/v
*:( 2Adam/conv1d_1/kernel/v
 : 2Adam/conv1d_1/bias/v
%:#
??2Adam/dense/kernel/v
:?2Adam/dense/bias/v
&:$	?2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_8888
?__inference_model_layer_call_and_return_conditional_losses_8035
?__inference_model_layer_call_and_return_conditional_losses_7987
?__inference_model_layer_call_and_return_conditional_losses_8579?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_7342?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
.?+
input_1??????????????????
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
?2?
$__inference_model_layer_call_fn_8919
$__inference_model_layer_call_fn_8194
$__inference_model_layer_call_fn_8115
$__inference_model_layer_call_fn_8950?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_8962
A__inference_dropout_layer_call_and_return_conditional_losses_8967?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_8972
&__inference_dropout_layer_call_fn_8977?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_9005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_graph_convolution_layer_call_fn_9013?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_9025
C__inference_dropout_1_layer_call_and_return_conditional_losses_9030?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_9040
(__inference_dropout_1_layer_call_fn_9035?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_9068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_graph_convolution_1_layer_call_fn_9076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_2_layer_call_and_return_conditional_losses_9093
C__inference_dropout_2_layer_call_and_return_conditional_losses_9088?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_2_layer_call_fn_9098
(__inference_dropout_2_layer_call_fn_9103?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_9131?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_graph_convolution_2_layer_call_fn_9139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_3_layer_call_and_return_conditional_losses_9151
C__inference_dropout_3_layer_call_and_return_conditional_losses_9156?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_3_layer_call_fn_9161
(__inference_dropout_3_layer_call_fn_9166?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_9194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_graph_convolution_3_layer_call_fn_9202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_sort_pooling_layer_call_and_return_conditional_losses_9369?
???
FullArgSpec)
args!?
jself
j
embeddings
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_sort_pooling_layer_call_fn_9375?
???
FullArgSpec)
args!?
jself
j
embeddings
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv1d_layer_call_and_return_conditional_losses_9390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv1d_layer_call_fn_9399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_7351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
,__inference_max_pooling1d_layer_call_fn_7357?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_9414?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv1d_1_layer_call_fn_9423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_9429?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_9434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_9445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_9454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_4_layer_call_and_return_conditional_losses_9466
C__inference_dropout_4_layer_call_and_return_conditional_losses_9471?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_4_layer_call_fn_9481
(__inference_dropout_4_layer_call_fn_9476?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_9492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_9501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_8235input_1input_2input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_7342?(1:DENOXYbc???
???
???
.?+
input_1??????????????????
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
? "1?.
,
dense_1!?
dense_1??????????
B__inference_conv1d_1_layer_call_and_return_conditional_losses_9414dNO3?0
)?&
$?!
inputs?????????
? ")?&
?
0????????? 
? ?
'__inference_conv1d_1_layer_call_fn_9423WNO3?0
)?&
$?!
inputs?????????
? "?????????? ?
@__inference_conv1d_layer_call_and_return_conditional_losses_9390eDE4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????#
? ?
%__inference_conv1d_layer_call_fn_9399XDE4?1
*?'
%?"
inputs??????????
? "??????????#?
A__inference_dense_1_layer_call_and_return_conditional_losses_9492]bc0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_dense_1_layer_call_fn_9501Pbc0?-
&?#
!?
inputs??????????
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_9445^XY0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? y
$__inference_dense_layer_call_fn_9454QXY0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_9025v@?=
6?3
-?*
inputs?????????????????? 
p
? "2?/
(?%
0?????????????????? 
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_9030v@?=
6?3
-?*
inputs?????????????????? 
p 
? "2?/
(?%
0?????????????????? 
? ?
(__inference_dropout_1_layer_call_fn_9035i@?=
6?3
-?*
inputs?????????????????? 
p
? "%?"?????????????????? ?
(__inference_dropout_1_layer_call_fn_9040i@?=
6?3
-?*
inputs?????????????????? 
p 
? "%?"?????????????????? ?
C__inference_dropout_2_layer_call_and_return_conditional_losses_9088v@?=
6?3
-?*
inputs?????????????????? 
p
? "2?/
(?%
0?????????????????? 
? ?
C__inference_dropout_2_layer_call_and_return_conditional_losses_9093v@?=
6?3
-?*
inputs?????????????????? 
p 
? "2?/
(?%
0?????????????????? 
? ?
(__inference_dropout_2_layer_call_fn_9098i@?=
6?3
-?*
inputs?????????????????? 
p
? "%?"?????????????????? ?
(__inference_dropout_2_layer_call_fn_9103i@?=
6?3
-?*
inputs?????????????????? 
p 
? "%?"?????????????????? ?
C__inference_dropout_3_layer_call_and_return_conditional_losses_9151v@?=
6?3
-?*
inputs?????????????????? 
p
? "2?/
(?%
0?????????????????? 
? ?
C__inference_dropout_3_layer_call_and_return_conditional_losses_9156v@?=
6?3
-?*
inputs?????????????????? 
p 
? "2?/
(?%
0?????????????????? 
? ?
(__inference_dropout_3_layer_call_fn_9161i@?=
6?3
-?*
inputs?????????????????? 
p
? "%?"?????????????????? ?
(__inference_dropout_3_layer_call_fn_9166i@?=
6?3
-?*
inputs?????????????????? 
p 
? "%?"?????????????????? ?
C__inference_dropout_4_layer_call_and_return_conditional_losses_9466^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
C__inference_dropout_4_layer_call_and_return_conditional_losses_9471^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? }
(__inference_dropout_4_layer_call_fn_9476Q4?1
*?'
!?
inputs??????????
p
? "???????????}
(__inference_dropout_4_layer_call_fn_9481Q4?1
*?'
!?
inputs??????????
p 
? "????????????
A__inference_dropout_layer_call_and_return_conditional_losses_8962v@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_8967v@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
&__inference_dropout_layer_call_fn_8972i@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
&__inference_dropout_layer_call_fn_8977i@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
A__inference_flatten_layer_call_and_return_conditional_losses_9429]3?0
)?&
$?!
inputs????????? 
? "&?#
?
0??????????
? z
&__inference_flatten_layer_call_fn_9434P3?0
)?&
$?!
inputs????????? 
? "????????????
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_9068?(}?z
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
? "2?/
(?%
0?????????????????? 
? ?
2__inference_graph_convolution_1_layer_call_fn_9076?(}?z
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
? "%?"?????????????????? ?
M__inference_graph_convolution_2_layer_call_and_return_conditional_losses_9131?1}?z
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
? "2?/
(?%
0?????????????????? 
? ?
2__inference_graph_convolution_2_layer_call_fn_9139?1}?z
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
? "%?"?????????????????? ?
M__inference_graph_convolution_3_layer_call_and_return_conditional_losses_9194?:}?z
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
? "2?/
(?%
0??????????????????
? ?
2__inference_graph_convolution_3_layer_call_fn_9202?:}?z
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
? "%?"???????????????????
K__inference_graph_convolution_layer_call_and_return_conditional_losses_9005?}?z
s?p
n?k
/?,
inputs/0??????????????????
8?5
inputs/1'???????????????????????????
? "2?/
(?%
0?????????????????? 
? ?
0__inference_graph_convolution_layer_call_fn_9013?}?z
s?p
n?k
/?,
inputs/0??????????????????
8?5
inputs/1'???????????????????????????
? "%?"?????????????????? ?
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_7351?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
,__inference_max_pooling1d_layer_call_fn_7357wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
?__inference_model_layer_call_and_return_conditional_losses_7987?(1:DENOXYbc???
???
???
.?+
input_1??????????????????
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8035?(1:DENOXYbc???
???
???
.?+
input_1??????????????????
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8579?(1:DENOXYbc???
???
???
/?,
inputs/0??????????????????
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8888?(1:DENOXYbc???
???
???
/?,
inputs/0??????????????????
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p 

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_8115?(1:DENOXYbc???
???
???
.?+
input_1??????????????????
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p

 
? "???????????
$__inference_model_layer_call_fn_8194?(1:DENOXYbc???
???
???
.?+
input_1??????????????????
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p 

 
? "???????????
$__inference_model_layer_call_fn_8919?(1:DENOXYbc???
???
???
/?,
inputs/0??????????????????
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p

 
? "???????????
$__inference_model_layer_call_fn_8950?(1:DENOXYbc???
???
???
/?,
inputs/0??????????????????
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p 

 
? "???????????
"__inference_signature_wrapper_8235?(1:DENOXYbc???
? 
???
9
input_1.?+
input_1??????????????????
5
input_2*?'
input_2??????????????????

B
input_37?4
input_3'???????????????????????????"1?.
,
dense_1!?
dense_1??????????
F__inference_sort_pooling_layer_call_and_return_conditional_losses_9369?i?f
_?\
1?.

embeddings??????????????????a
'?$
mask??????????????????

? "*?'
 ?
0??????????
? ?
+__inference_sort_pooling_layer_call_fn_9375?i?f
_?\
1?.

embeddings??????????????????a
'?$
mask??????????????????

? "???????????