Ù¶
Ñ¢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68

|
chroma_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namechroma_1/kernel
u
#chroma_1/kernel/Read/ReadVariableOpReadVariableOpchroma_1/kernel* 
_output_shapes
:
*
dtype0
s
chroma_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechroma_1/bias
l
!chroma_1/bias/Read/ReadVariableOpReadVariableOpchroma_1/bias*
_output_shapes	
:*
dtype0
|
chroma_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namechroma_2/kernel
u
#chroma_2/kernel/Read/ReadVariableOpReadVariableOpchroma_2/kernel* 
_output_shapes
:
*
dtype0
s
chroma_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechroma_2/bias
l
!chroma_2/bias/Read/ReadVariableOpReadVariableOpchroma_2/bias*
_output_shapes	
:*
dtype0
|
chroma_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namechroma_3/kernel
u
#chroma_3/kernel/Read/ReadVariableOpReadVariableOpchroma_3/kernel* 
_output_shapes
:
*
dtype0
s
chroma_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechroma_3/bias
l
!chroma_3/bias/Read/ReadVariableOpReadVariableOpchroma_3/bias*
_output_shapes	
:*
dtype0
|
chroma_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namechroma_4/kernel
u
#chroma_4/kernel/Read/ReadVariableOpReadVariableOpchroma_4/kernel* 
_output_shapes
:
*
dtype0
s
chroma_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechroma_4/bias
l
!chroma_4/bias/Read/ReadVariableOpReadVariableOpchroma_4/bias*
_output_shapes	
:*
dtype0
{
chroma_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namechroma_5/kernel
t
#chroma_5/kernel/Read/ReadVariableOpReadVariableOpchroma_5/kernel*
_output_shapes
:	@*
dtype0
r
chroma_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namechroma_5/bias
k
!chroma_5/bias/Read/ReadVariableOpReadVariableOpchroma_5/bias*
_output_shapes
:@*
dtype0
z
chroma_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namechroma_6/kernel
s
#chroma_6/kernel/Read/ReadVariableOpReadVariableOpchroma_6/kernel*
_output_shapes

:@ *
dtype0
r
chroma_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namechroma_6/bias
k
!chroma_6/bias/Read/ReadVariableOpReadVariableOpchroma_6/bias*
_output_shapes
: *
dtype0
z
chroma_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namechroma_7/kernel
s
#chroma_7/kernel/Read/ReadVariableOpReadVariableOpchroma_7/kernel*
_output_shapes

:  *
dtype0
r
chroma_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namechroma_7/bias
k
!chroma_7/bias/Read/ReadVariableOpReadVariableOpchroma_7/bias*
_output_shapes
: *
dtype0

chroma_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_namechroma_output/kernel
}
(chroma_output/kernel/Read/ReadVariableOpReadVariableOpchroma_output/kernel*
_output_shapes

: *
dtype0
|
chroma_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namechroma_output/bias
u
&chroma_output/bias/Read/ReadVariableOpReadVariableOpchroma_output/bias*
_output_shapes
:*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/chroma_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_1/kernel/m

*Adam/chroma_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/chroma_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_1/bias/m
z
(Adam/chroma_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_1/bias/m*
_output_shapes	
:*
dtype0

Adam/chroma_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_2/kernel/m

*Adam/chroma_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/chroma_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_2/bias/m
z
(Adam/chroma_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_2/bias/m*
_output_shapes	
:*
dtype0

Adam/chroma_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_3/kernel/m

*Adam/chroma_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/chroma_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_3/bias/m
z
(Adam/chroma_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_3/bias/m*
_output_shapes	
:*
dtype0

Adam/chroma_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_4/kernel/m

*Adam/chroma_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_4/kernel/m* 
_output_shapes
:
*
dtype0

Adam/chroma_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_4/bias/m
z
(Adam/chroma_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_4/bias/m*
_output_shapes	
:*
dtype0

Adam/chroma_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/chroma_5/kernel/m

*Adam/chroma_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_5/kernel/m*
_output_shapes
:	@*
dtype0

Adam/chroma_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/chroma_5/bias/m
y
(Adam/chroma_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_5/bias/m*
_output_shapes
:@*
dtype0

Adam/chroma_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/chroma_6/kernel/m

*Adam/chroma_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_6/kernel/m*
_output_shapes

:@ *
dtype0

Adam/chroma_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/chroma_6/bias/m
y
(Adam/chroma_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_6/bias/m*
_output_shapes
: *
dtype0

Adam/chroma_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/chroma_7/kernel/m

*Adam/chroma_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_7/kernel/m*
_output_shapes

:  *
dtype0

Adam/chroma_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/chroma_7/bias/m
y
(Adam/chroma_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_7/bias/m*
_output_shapes
: *
dtype0

Adam/chroma_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_nameAdam/chroma_output/kernel/m

/Adam/chroma_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chroma_output/kernel/m*
_output_shapes

: *
dtype0

Adam/chroma_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/chroma_output/bias/m

-Adam/chroma_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/chroma_output/bias/m*
_output_shapes
:*
dtype0

Adam/chroma_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_1/kernel/v

*Adam/chroma_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/chroma_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_1/bias/v
z
(Adam/chroma_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_1/bias/v*
_output_shapes	
:*
dtype0

Adam/chroma_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_2/kernel/v

*Adam/chroma_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/chroma_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_2/bias/v
z
(Adam/chroma_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_2/bias/v*
_output_shapes	
:*
dtype0

Adam/chroma_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_3/kernel/v

*Adam/chroma_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/chroma_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_3/bias/v
z
(Adam/chroma_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_3/bias/v*
_output_shapes	
:*
dtype0

Adam/chroma_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/chroma_4/kernel/v

*Adam/chroma_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_4/kernel/v* 
_output_shapes
:
*
dtype0

Adam/chroma_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/chroma_4/bias/v
z
(Adam/chroma_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_4/bias/v*
_output_shapes	
:*
dtype0

Adam/chroma_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/chroma_5/kernel/v

*Adam/chroma_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_5/kernel/v*
_output_shapes
:	@*
dtype0

Adam/chroma_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/chroma_5/bias/v
y
(Adam/chroma_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_5/bias/v*
_output_shapes
:@*
dtype0

Adam/chroma_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/chroma_6/kernel/v

*Adam/chroma_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_6/kernel/v*
_output_shapes

:@ *
dtype0

Adam/chroma_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/chroma_6/bias/v
y
(Adam/chroma_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_6/bias/v*
_output_shapes
: *
dtype0

Adam/chroma_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/chroma_7/kernel/v

*Adam/chroma_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_7/kernel/v*
_output_shapes

:  *
dtype0

Adam/chroma_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/chroma_7/bias/v
y
(Adam/chroma_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_7/bias/v*
_output_shapes
: *
dtype0

Adam/chroma_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_nameAdam/chroma_output/kernel/v

/Adam/chroma_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chroma_output/kernel/v*
_output_shapes

: *
dtype0

Adam/chroma_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/chroma_output/bias/v

-Adam/chroma_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/chroma_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Öd
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*d
valuedBd Býc

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
¦

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
¦

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
¦

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
¦

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses*
¦

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
¦

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses*

Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemmmm"m#m*m+m2m3m:m ;m¡Bm¢Cm£Jm¤Km¥v¦v§v¨v©"vª#v«*v¬+v­2v®3v¯:v°;v±Bv²Cv³Jv´Kvµ*
z
0
1
2
3
"4
#5
*6
+7
28
39
:10
;11
B12
C13
J14
K15*
z
0
1
2
3
"4
#5
*6
+7
28
39
:10
;11
B12
C13
J14
K15*
* 
°
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

\serving_default* 
_Y
VARIABLE_VALUEchroma_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEchroma_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEchroma_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEchroma_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 

lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEchroma_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEchroma_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

:0
;1*

:0
;1*
* 

vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEchroma_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchroma_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

B0
C1*
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
d^
VARIABLE_VALUEchroma_output/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEchroma_output/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

J0
K1*

J0
K1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
|
VARIABLE_VALUEAdam/chroma_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/chroma_output/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/chroma_output/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/chroma_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/chroma_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/chroma_output/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/chroma_output/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_chroma_1_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ä
StatefulPartitionedCallStatefulPartitionedCallserving_default_chroma_1_inputchroma_1/kernelchroma_1/biaschroma_2/kernelchroma_2/biaschroma_3/kernelchroma_3/biaschroma_4/kernelchroma_4/biaschroma_5/kernelchroma_5/biaschroma_6/kernelchroma_6/biaschroma_7/kernelchroma_7/biaschroma_output/kernelchroma_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_4630600
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
è
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#chroma_1/kernel/Read/ReadVariableOp!chroma_1/bias/Read/ReadVariableOp#chroma_2/kernel/Read/ReadVariableOp!chroma_2/bias/Read/ReadVariableOp#chroma_3/kernel/Read/ReadVariableOp!chroma_3/bias/Read/ReadVariableOp#chroma_4/kernel/Read/ReadVariableOp!chroma_4/bias/Read/ReadVariableOp#chroma_5/kernel/Read/ReadVariableOp!chroma_5/bias/Read/ReadVariableOp#chroma_6/kernel/Read/ReadVariableOp!chroma_6/bias/Read/ReadVariableOp#chroma_7/kernel/Read/ReadVariableOp!chroma_7/bias/Read/ReadVariableOp(chroma_output/kernel/Read/ReadVariableOp&chroma_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/chroma_1/kernel/m/Read/ReadVariableOp(Adam/chroma_1/bias/m/Read/ReadVariableOp*Adam/chroma_2/kernel/m/Read/ReadVariableOp(Adam/chroma_2/bias/m/Read/ReadVariableOp*Adam/chroma_3/kernel/m/Read/ReadVariableOp(Adam/chroma_3/bias/m/Read/ReadVariableOp*Adam/chroma_4/kernel/m/Read/ReadVariableOp(Adam/chroma_4/bias/m/Read/ReadVariableOp*Adam/chroma_5/kernel/m/Read/ReadVariableOp(Adam/chroma_5/bias/m/Read/ReadVariableOp*Adam/chroma_6/kernel/m/Read/ReadVariableOp(Adam/chroma_6/bias/m/Read/ReadVariableOp*Adam/chroma_7/kernel/m/Read/ReadVariableOp(Adam/chroma_7/bias/m/Read/ReadVariableOp/Adam/chroma_output/kernel/m/Read/ReadVariableOp-Adam/chroma_output/bias/m/Read/ReadVariableOp*Adam/chroma_1/kernel/v/Read/ReadVariableOp(Adam/chroma_1/bias/v/Read/ReadVariableOp*Adam/chroma_2/kernel/v/Read/ReadVariableOp(Adam/chroma_2/bias/v/Read/ReadVariableOp*Adam/chroma_3/kernel/v/Read/ReadVariableOp(Adam/chroma_3/bias/v/Read/ReadVariableOp*Adam/chroma_4/kernel/v/Read/ReadVariableOp(Adam/chroma_4/bias/v/Read/ReadVariableOp*Adam/chroma_5/kernel/v/Read/ReadVariableOp(Adam/chroma_5/bias/v/Read/ReadVariableOp*Adam/chroma_6/kernel/v/Read/ReadVariableOp(Adam/chroma_6/bias/v/Read/ReadVariableOp*Adam/chroma_7/kernel/v/Read/ReadVariableOp(Adam/chroma_7/bias/v/Read/ReadVariableOp/Adam/chroma_output/kernel/v/Read/ReadVariableOp-Adam/chroma_output/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_4630960
Ç
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamechroma_1/kernelchroma_1/biaschroma_2/kernelchroma_2/biaschroma_3/kernelchroma_3/biaschroma_4/kernelchroma_4/biaschroma_5/kernelchroma_5/biaschroma_6/kernelchroma_6/biaschroma_7/kernelchroma_7/biaschroma_output/kernelchroma_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/chroma_1/kernel/mAdam/chroma_1/bias/mAdam/chroma_2/kernel/mAdam/chroma_2/bias/mAdam/chroma_3/kernel/mAdam/chroma_3/bias/mAdam/chroma_4/kernel/mAdam/chroma_4/bias/mAdam/chroma_5/kernel/mAdam/chroma_5/bias/mAdam/chroma_6/kernel/mAdam/chroma_6/bias/mAdam/chroma_7/kernel/mAdam/chroma_7/bias/mAdam/chroma_output/kernel/mAdam/chroma_output/bias/mAdam/chroma_1/kernel/vAdam/chroma_1/bias/vAdam/chroma_2/kernel/vAdam/chroma_2/bias/vAdam/chroma_3/kernel/vAdam/chroma_3/bias/vAdam/chroma_4/kernel/vAdam/chroma_4/bias/vAdam/chroma_5/kernel/vAdam/chroma_5/bias/vAdam/chroma_6/kernel/vAdam/chroma_6/bias/vAdam/chroma_7/kernel/vAdam/chroma_7/bias/vAdam/chroma_output/kernel/vAdam/chroma_output/bias/v*G
Tin@
>2<*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_4631147ï


ö
E__inference_chroma_6_layer_call_and_return_conditional_losses_4629962

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÕU
ú
"__inference__wrapped_model_4629859
chroma_1_inputF
2sequential_chroma_1_matmul_readvariableop_resource:
B
3sequential_chroma_1_biasadd_readvariableop_resource:	F
2sequential_chroma_2_matmul_readvariableop_resource:
B
3sequential_chroma_2_biasadd_readvariableop_resource:	F
2sequential_chroma_3_matmul_readvariableop_resource:
B
3sequential_chroma_3_biasadd_readvariableop_resource:	F
2sequential_chroma_4_matmul_readvariableop_resource:
B
3sequential_chroma_4_biasadd_readvariableop_resource:	E
2sequential_chroma_5_matmul_readvariableop_resource:	@A
3sequential_chroma_5_biasadd_readvariableop_resource:@D
2sequential_chroma_6_matmul_readvariableop_resource:@ A
3sequential_chroma_6_biasadd_readvariableop_resource: D
2sequential_chroma_7_matmul_readvariableop_resource:  A
3sequential_chroma_7_biasadd_readvariableop_resource: I
7sequential_chroma_output_matmul_readvariableop_resource: F
8sequential_chroma_output_biasadd_readvariableop_resource:
identity¢*sequential/chroma_1/BiasAdd/ReadVariableOp¢)sequential/chroma_1/MatMul/ReadVariableOp¢*sequential/chroma_2/BiasAdd/ReadVariableOp¢)sequential/chroma_2/MatMul/ReadVariableOp¢*sequential/chroma_3/BiasAdd/ReadVariableOp¢)sequential/chroma_3/MatMul/ReadVariableOp¢*sequential/chroma_4/BiasAdd/ReadVariableOp¢)sequential/chroma_4/MatMul/ReadVariableOp¢*sequential/chroma_5/BiasAdd/ReadVariableOp¢)sequential/chroma_5/MatMul/ReadVariableOp¢*sequential/chroma_6/BiasAdd/ReadVariableOp¢)sequential/chroma_6/MatMul/ReadVariableOp¢*sequential/chroma_7/BiasAdd/ReadVariableOp¢)sequential/chroma_7/MatMul/ReadVariableOp¢/sequential/chroma_output/BiasAdd/ReadVariableOp¢.sequential/chroma_output/MatMul/ReadVariableOp
)sequential/chroma_1/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sequential/chroma_1/MatMulMatMulchroma_1_input1sequential/chroma_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential/chroma_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0³
sequential/chroma_1/BiasAddBiasAdd$sequential/chroma_1/MatMul:product:02sequential/chroma_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
sequential/chroma_1/ReluRelu$sequential/chroma_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/chroma_2/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0²
sequential/chroma_2/MatMulMatMul&sequential/chroma_1/Relu:activations:01sequential/chroma_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential/chroma_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0³
sequential/chroma_2/BiasAddBiasAdd$sequential/chroma_2/MatMul:product:02sequential/chroma_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
sequential/chroma_2/ReluRelu$sequential/chroma_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/chroma_3/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0²
sequential/chroma_3/MatMulMatMul&sequential/chroma_2/Relu:activations:01sequential/chroma_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential/chroma_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0³
sequential/chroma_3/BiasAddBiasAdd$sequential/chroma_3/MatMul:product:02sequential/chroma_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
sequential/chroma_3/ReluRelu$sequential/chroma_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/chroma_4/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0²
sequential/chroma_4/MatMulMatMul&sequential/chroma_3/Relu:activations:01sequential/chroma_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential/chroma_4/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0³
sequential/chroma_4/BiasAddBiasAdd$sequential/chroma_4/MatMul:product:02sequential/chroma_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
sequential/chroma_4/ReluRelu$sequential/chroma_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/chroma_5/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_5_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0±
sequential/chroma_5/MatMulMatMul&sequential/chroma_4/Relu:activations:01sequential/chroma_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential/chroma_5/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0²
sequential/chroma_5/BiasAddBiasAdd$sequential/chroma_5/MatMul:product:02sequential/chroma_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
sequential/chroma_5/ReluRelu$sequential/chroma_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)sequential/chroma_6/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0±
sequential/chroma_6/MatMulMatMul&sequential/chroma_5/Relu:activations:01sequential/chroma_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential/chroma_6/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0²
sequential/chroma_6/BiasAddBiasAdd$sequential/chroma_6/MatMul:product:02sequential/chroma_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
sequential/chroma_6/ReluRelu$sequential/chroma_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/chroma_7/MatMul/ReadVariableOpReadVariableOp2sequential_chroma_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0±
sequential/chroma_7/MatMulMatMul&sequential/chroma_6/Relu:activations:01sequential/chroma_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential/chroma_7/BiasAdd/ReadVariableOpReadVariableOp3sequential_chroma_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0²
sequential/chroma_7/BiasAddBiasAdd$sequential/chroma_7/MatMul:product:02sequential/chroma_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
sequential/chroma_7/ReluRelu$sequential/chroma_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
.sequential/chroma_output/MatMul/ReadVariableOpReadVariableOp7sequential_chroma_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0»
sequential/chroma_output/MatMulMatMul&sequential/chroma_7/Relu:activations:06sequential/chroma_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential/chroma_output/BiasAdd/ReadVariableOpReadVariableOp8sequential_chroma_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential/chroma_output/BiasAddBiasAdd)sequential/chroma_output/MatMul:product:07sequential/chroma_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/chroma_output/ReluRelu)sequential/chroma_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity+sequential/chroma_output/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp+^sequential/chroma_1/BiasAdd/ReadVariableOp*^sequential/chroma_1/MatMul/ReadVariableOp+^sequential/chroma_2/BiasAdd/ReadVariableOp*^sequential/chroma_2/MatMul/ReadVariableOp+^sequential/chroma_3/BiasAdd/ReadVariableOp*^sequential/chroma_3/MatMul/ReadVariableOp+^sequential/chroma_4/BiasAdd/ReadVariableOp*^sequential/chroma_4/MatMul/ReadVariableOp+^sequential/chroma_5/BiasAdd/ReadVariableOp*^sequential/chroma_5/MatMul/ReadVariableOp+^sequential/chroma_6/BiasAdd/ReadVariableOp*^sequential/chroma_6/MatMul/ReadVariableOp+^sequential/chroma_7/BiasAdd/ReadVariableOp*^sequential/chroma_7/MatMul/ReadVariableOp0^sequential/chroma_output/BiasAdd/ReadVariableOp/^sequential/chroma_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2X
*sequential/chroma_1/BiasAdd/ReadVariableOp*sequential/chroma_1/BiasAdd/ReadVariableOp2V
)sequential/chroma_1/MatMul/ReadVariableOp)sequential/chroma_1/MatMul/ReadVariableOp2X
*sequential/chroma_2/BiasAdd/ReadVariableOp*sequential/chroma_2/BiasAdd/ReadVariableOp2V
)sequential/chroma_2/MatMul/ReadVariableOp)sequential/chroma_2/MatMul/ReadVariableOp2X
*sequential/chroma_3/BiasAdd/ReadVariableOp*sequential/chroma_3/BiasAdd/ReadVariableOp2V
)sequential/chroma_3/MatMul/ReadVariableOp)sequential/chroma_3/MatMul/ReadVariableOp2X
*sequential/chroma_4/BiasAdd/ReadVariableOp*sequential/chroma_4/BiasAdd/ReadVariableOp2V
)sequential/chroma_4/MatMul/ReadVariableOp)sequential/chroma_4/MatMul/ReadVariableOp2X
*sequential/chroma_5/BiasAdd/ReadVariableOp*sequential/chroma_5/BiasAdd/ReadVariableOp2V
)sequential/chroma_5/MatMul/ReadVariableOp)sequential/chroma_5/MatMul/ReadVariableOp2X
*sequential/chroma_6/BiasAdd/ReadVariableOp*sequential/chroma_6/BiasAdd/ReadVariableOp2V
)sequential/chroma_6/MatMul/ReadVariableOp)sequential/chroma_6/MatMul/ReadVariableOp2X
*sequential/chroma_7/BiasAdd/ReadVariableOp*sequential/chroma_7/BiasAdd/ReadVariableOp2V
)sequential/chroma_7/MatMul/ReadVariableOp)sequential/chroma_7/MatMul/ReadVariableOp2b
/sequential/chroma_output/BiasAdd/ReadVariableOp/sequential/chroma_output/BiasAdd/ReadVariableOp2`
.sequential/chroma_output/MatMul/ReadVariableOp.sequential/chroma_output/MatMul/ReadVariableOp:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namechroma_1_input
+
Â
G__inference_sequential_layer_call_and_return_conditional_losses_4630317
chroma_1_input$
chroma_1_4630276:

chroma_1_4630278:	$
chroma_2_4630281:

chroma_2_4630283:	$
chroma_3_4630286:

chroma_3_4630288:	$
chroma_4_4630291:

chroma_4_4630293:	#
chroma_5_4630296:	@
chroma_5_4630298:@"
chroma_6_4630301:@ 
chroma_6_4630303: "
chroma_7_4630306:  
chroma_7_4630308: '
chroma_output_4630311: #
chroma_output_4630313:
identity¢ chroma_1/StatefulPartitionedCall¢ chroma_2/StatefulPartitionedCall¢ chroma_3/StatefulPartitionedCall¢ chroma_4/StatefulPartitionedCall¢ chroma_5/StatefulPartitionedCall¢ chroma_6/StatefulPartitionedCall¢ chroma_7/StatefulPartitionedCall¢%chroma_output/StatefulPartitionedCallü
 chroma_1/StatefulPartitionedCallStatefulPartitionedCallchroma_1_inputchroma_1_4630276chroma_1_4630278*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_1_layer_call_and_return_conditional_losses_4629877
 chroma_2/StatefulPartitionedCallStatefulPartitionedCall)chroma_1/StatefulPartitionedCall:output:0chroma_2_4630281chroma_2_4630283*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_2_layer_call_and_return_conditional_losses_4629894
 chroma_3/StatefulPartitionedCallStatefulPartitionedCall)chroma_2/StatefulPartitionedCall:output:0chroma_3_4630286chroma_3_4630288*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_3_layer_call_and_return_conditional_losses_4629911
 chroma_4/StatefulPartitionedCallStatefulPartitionedCall)chroma_3/StatefulPartitionedCall:output:0chroma_4_4630291chroma_4_4630293*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_4_layer_call_and_return_conditional_losses_4629928
 chroma_5/StatefulPartitionedCallStatefulPartitionedCall)chroma_4/StatefulPartitionedCall:output:0chroma_5_4630296chroma_5_4630298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_5_layer_call_and_return_conditional_losses_4629945
 chroma_6/StatefulPartitionedCallStatefulPartitionedCall)chroma_5/StatefulPartitionedCall:output:0chroma_6_4630301chroma_6_4630303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_6_layer_call_and_return_conditional_losses_4629962
 chroma_7/StatefulPartitionedCallStatefulPartitionedCall)chroma_6/StatefulPartitionedCall:output:0chroma_7_4630306chroma_7_4630308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_7_layer_call_and_return_conditional_losses_4629979ª
%chroma_output/StatefulPartitionedCallStatefulPartitionedCall)chroma_7/StatefulPartitionedCall:output:0chroma_output_4630311chroma_output_4630313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_chroma_output_layer_call_and_return_conditional_losses_4629996}
IdentityIdentity.chroma_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp!^chroma_1/StatefulPartitionedCall!^chroma_2/StatefulPartitionedCall!^chroma_3/StatefulPartitionedCall!^chroma_4/StatefulPartitionedCall!^chroma_5/StatefulPartitionedCall!^chroma_6/StatefulPartitionedCall!^chroma_7/StatefulPartitionedCall&^chroma_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 chroma_1/StatefulPartitionedCall chroma_1/StatefulPartitionedCall2D
 chroma_2/StatefulPartitionedCall chroma_2/StatefulPartitionedCall2D
 chroma_3/StatefulPartitionedCall chroma_3/StatefulPartitionedCall2D
 chroma_4/StatefulPartitionedCall chroma_4/StatefulPartitionedCall2D
 chroma_5/StatefulPartitionedCall chroma_5/StatefulPartitionedCall2D
 chroma_6/StatefulPartitionedCall chroma_6/StatefulPartitionedCall2D
 chroma_7/StatefulPartitionedCall chroma_7/StatefulPartitionedCall2N
%chroma_output/StatefulPartitionedCall%chroma_output/StatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namechroma_1_input
¡

û
J__inference_chroma_output_layer_call_and_return_conditional_losses_4629996

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¦
%__inference_signature_wrapper_4630600
chroma_1_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallchroma_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_4629859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namechroma_1_input
ñé
ê#
#__inference__traced_restore_4631147
file_prefix4
 assignvariableop_chroma_1_kernel:
/
 assignvariableop_1_chroma_1_bias:	6
"assignvariableop_2_chroma_2_kernel:
/
 assignvariableop_3_chroma_2_bias:	6
"assignvariableop_4_chroma_3_kernel:
/
 assignvariableop_5_chroma_3_bias:	6
"assignvariableop_6_chroma_4_kernel:
/
 assignvariableop_7_chroma_4_bias:	5
"assignvariableop_8_chroma_5_kernel:	@.
 assignvariableop_9_chroma_5_bias:@5
#assignvariableop_10_chroma_6_kernel:@ /
!assignvariableop_11_chroma_6_bias: 5
#assignvariableop_12_chroma_7_kernel:  /
!assignvariableop_13_chroma_7_bias: :
(assignvariableop_14_chroma_output_kernel: 4
&assignvariableop_15_chroma_output_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: %
assignvariableop_25_total_2: %
assignvariableop_26_count_2: >
*assignvariableop_27_adam_chroma_1_kernel_m:
7
(assignvariableop_28_adam_chroma_1_bias_m:	>
*assignvariableop_29_adam_chroma_2_kernel_m:
7
(assignvariableop_30_adam_chroma_2_bias_m:	>
*assignvariableop_31_adam_chroma_3_kernel_m:
7
(assignvariableop_32_adam_chroma_3_bias_m:	>
*assignvariableop_33_adam_chroma_4_kernel_m:
7
(assignvariableop_34_adam_chroma_4_bias_m:	=
*assignvariableop_35_adam_chroma_5_kernel_m:	@6
(assignvariableop_36_adam_chroma_5_bias_m:@<
*assignvariableop_37_adam_chroma_6_kernel_m:@ 6
(assignvariableop_38_adam_chroma_6_bias_m: <
*assignvariableop_39_adam_chroma_7_kernel_m:  6
(assignvariableop_40_adam_chroma_7_bias_m: A
/assignvariableop_41_adam_chroma_output_kernel_m: ;
-assignvariableop_42_adam_chroma_output_bias_m:>
*assignvariableop_43_adam_chroma_1_kernel_v:
7
(assignvariableop_44_adam_chroma_1_bias_v:	>
*assignvariableop_45_adam_chroma_2_kernel_v:
7
(assignvariableop_46_adam_chroma_2_bias_v:	>
*assignvariableop_47_adam_chroma_3_kernel_v:
7
(assignvariableop_48_adam_chroma_3_bias_v:	>
*assignvariableop_49_adam_chroma_4_kernel_v:
7
(assignvariableop_50_adam_chroma_4_bias_v:	=
*assignvariableop_51_adam_chroma_5_kernel_v:	@6
(assignvariableop_52_adam_chroma_5_bias_v:@<
*assignvariableop_53_adam_chroma_6_kernel_v:@ 6
(assignvariableop_54_adam_chroma_6_bias_v: <
*assignvariableop_55_adam_chroma_7_kernel_v:  6
(assignvariableop_56_adam_chroma_7_bias_v: A
/assignvariableop_57_adam_chroma_output_kernel_v: ;
-assignvariableop_58_adam_chroma_output_bias_v:
identity_60¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*¬ 
value¢ B <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHë
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Í
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesó
ð::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_chroma_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_chroma_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_chroma_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_chroma_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_chroma_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_chroma_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_chroma_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_chroma_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_chroma_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_chroma_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_chroma_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_chroma_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_chroma_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_chroma_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp(assignvariableop_14_chroma_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp&assignvariableop_15_chroma_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_chroma_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_chroma_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_chroma_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_chroma_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_chroma_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_chroma_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_chroma_4_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_chroma_4_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_chroma_5_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_chroma_5_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_chroma_6_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_chroma_6_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_chroma_7_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_chroma_7_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_41AssignVariableOp/assignvariableop_41_adam_chroma_output_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp-assignvariableop_42_adam_chroma_output_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_chroma_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_chroma_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_chroma_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_chroma_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_chroma_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_chroma_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_chroma_4_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_chroma_4_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_chroma_5_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_chroma_5_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_chroma_6_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_chroma_6_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_chroma_7_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_chroma_7_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_57AssignVariableOp/assignvariableop_57_adam_chroma_output_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp-assignvariableop_58_adam_chroma_output_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 á

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_60IdentityIdentity_59:output:0^NoOp_1*
T0*
_output_shapes
: Î

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_60Identity_60:output:0*
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ý*
º
G__inference_sequential_layer_call_and_return_conditional_losses_4630003

inputs$
chroma_1_4629878:

chroma_1_4629880:	$
chroma_2_4629895:

chroma_2_4629897:	$
chroma_3_4629912:

chroma_3_4629914:	$
chroma_4_4629929:

chroma_4_4629931:	#
chroma_5_4629946:	@
chroma_5_4629948:@"
chroma_6_4629963:@ 
chroma_6_4629965: "
chroma_7_4629980:  
chroma_7_4629982: '
chroma_output_4629997: #
chroma_output_4629999:
identity¢ chroma_1/StatefulPartitionedCall¢ chroma_2/StatefulPartitionedCall¢ chroma_3/StatefulPartitionedCall¢ chroma_4/StatefulPartitionedCall¢ chroma_5/StatefulPartitionedCall¢ chroma_6/StatefulPartitionedCall¢ chroma_7/StatefulPartitionedCall¢%chroma_output/StatefulPartitionedCallô
 chroma_1/StatefulPartitionedCallStatefulPartitionedCallinputschroma_1_4629878chroma_1_4629880*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_1_layer_call_and_return_conditional_losses_4629877
 chroma_2/StatefulPartitionedCallStatefulPartitionedCall)chroma_1/StatefulPartitionedCall:output:0chroma_2_4629895chroma_2_4629897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_2_layer_call_and_return_conditional_losses_4629894
 chroma_3/StatefulPartitionedCallStatefulPartitionedCall)chroma_2/StatefulPartitionedCall:output:0chroma_3_4629912chroma_3_4629914*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_3_layer_call_and_return_conditional_losses_4629911
 chroma_4/StatefulPartitionedCallStatefulPartitionedCall)chroma_3/StatefulPartitionedCall:output:0chroma_4_4629929chroma_4_4629931*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_4_layer_call_and_return_conditional_losses_4629928
 chroma_5/StatefulPartitionedCallStatefulPartitionedCall)chroma_4/StatefulPartitionedCall:output:0chroma_5_4629946chroma_5_4629948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_5_layer_call_and_return_conditional_losses_4629945
 chroma_6/StatefulPartitionedCallStatefulPartitionedCall)chroma_5/StatefulPartitionedCall:output:0chroma_6_4629963chroma_6_4629965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_6_layer_call_and_return_conditional_losses_4629962
 chroma_7/StatefulPartitionedCallStatefulPartitionedCall)chroma_6/StatefulPartitionedCall:output:0chroma_7_4629980chroma_7_4629982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_7_layer_call_and_return_conditional_losses_4629979ª
%chroma_output/StatefulPartitionedCallStatefulPartitionedCall)chroma_7/StatefulPartitionedCall:output:0chroma_output_4629997chroma_output_4629999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_chroma_output_layer_call_and_return_conditional_losses_4629996}
IdentityIdentity.chroma_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp!^chroma_1/StatefulPartitionedCall!^chroma_2/StatefulPartitionedCall!^chroma_3/StatefulPartitionedCall!^chroma_4/StatefulPartitionedCall!^chroma_5/StatefulPartitionedCall!^chroma_6/StatefulPartitionedCall!^chroma_7/StatefulPartitionedCall&^chroma_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 chroma_1/StatefulPartitionedCall chroma_1/StatefulPartitionedCall2D
 chroma_2/StatefulPartitionedCall chroma_2/StatefulPartitionedCall2D
 chroma_3/StatefulPartitionedCall chroma_3/StatefulPartitionedCall2D
 chroma_4/StatefulPartitionedCall chroma_4/StatefulPartitionedCall2D
 chroma_5/StatefulPartitionedCall chroma_5/StatefulPartitionedCall2D
 chroma_6/StatefulPartitionedCall chroma_6/StatefulPartitionedCall2D
 chroma_7/StatefulPartitionedCall chroma_7/StatefulPartitionedCall2N
%chroma_output/StatefulPartitionedCall%chroma_output/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_3_layer_call_and_return_conditional_losses_4630660

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_chroma_6_layer_call_and_return_conditional_losses_4630720

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¡

û
J__inference_chroma_output_layer_call_and_return_conditional_losses_4630760

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_4_layer_call_and_return_conditional_losses_4629928

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_chroma_7_layer_call_fn_4630729

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_7_layer_call_and_return_conditional_losses_4629979o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÐF
·
G__inference_sequential_layer_call_and_return_conditional_losses_4630501

inputs;
'chroma_1_matmul_readvariableop_resource:
7
(chroma_1_biasadd_readvariableop_resource:	;
'chroma_2_matmul_readvariableop_resource:
7
(chroma_2_biasadd_readvariableop_resource:	;
'chroma_3_matmul_readvariableop_resource:
7
(chroma_3_biasadd_readvariableop_resource:	;
'chroma_4_matmul_readvariableop_resource:
7
(chroma_4_biasadd_readvariableop_resource:	:
'chroma_5_matmul_readvariableop_resource:	@6
(chroma_5_biasadd_readvariableop_resource:@9
'chroma_6_matmul_readvariableop_resource:@ 6
(chroma_6_biasadd_readvariableop_resource: 9
'chroma_7_matmul_readvariableop_resource:  6
(chroma_7_biasadd_readvariableop_resource: >
,chroma_output_matmul_readvariableop_resource: ;
-chroma_output_biasadd_readvariableop_resource:
identity¢chroma_1/BiasAdd/ReadVariableOp¢chroma_1/MatMul/ReadVariableOp¢chroma_2/BiasAdd/ReadVariableOp¢chroma_2/MatMul/ReadVariableOp¢chroma_3/BiasAdd/ReadVariableOp¢chroma_3/MatMul/ReadVariableOp¢chroma_4/BiasAdd/ReadVariableOp¢chroma_4/MatMul/ReadVariableOp¢chroma_5/BiasAdd/ReadVariableOp¢chroma_5/MatMul/ReadVariableOp¢chroma_6/BiasAdd/ReadVariableOp¢chroma_6/MatMul/ReadVariableOp¢chroma_7/BiasAdd/ReadVariableOp¢chroma_7/MatMul/ReadVariableOp¢$chroma_output/BiasAdd/ReadVariableOp¢#chroma_output/MatMul/ReadVariableOp
chroma_1/MatMul/ReadVariableOpReadVariableOp'chroma_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0|
chroma_1/MatMulMatMulinputs&chroma_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_1/BiasAdd/ReadVariableOpReadVariableOp(chroma_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_1/BiasAddBiasAddchroma_1/MatMul:product:0'chroma_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_1/ReluReluchroma_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_2/MatMul/ReadVariableOpReadVariableOp'chroma_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
chroma_2/MatMulMatMulchroma_1/Relu:activations:0&chroma_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_2/BiasAdd/ReadVariableOpReadVariableOp(chroma_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_2/BiasAddBiasAddchroma_2/MatMul:product:0'chroma_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_2/ReluReluchroma_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_3/MatMul/ReadVariableOpReadVariableOp'chroma_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
chroma_3/MatMulMatMulchroma_2/Relu:activations:0&chroma_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_3/BiasAdd/ReadVariableOpReadVariableOp(chroma_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_3/BiasAddBiasAddchroma_3/MatMul:product:0'chroma_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_3/ReluReluchroma_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_4/MatMul/ReadVariableOpReadVariableOp'chroma_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
chroma_4/MatMulMatMulchroma_3/Relu:activations:0&chroma_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_4/BiasAdd/ReadVariableOpReadVariableOp(chroma_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_4/BiasAddBiasAddchroma_4/MatMul:product:0'chroma_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_4/ReluReluchroma_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_5/MatMul/ReadVariableOpReadVariableOp'chroma_5_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
chroma_5/MatMulMatMulchroma_4/Relu:activations:0&chroma_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
chroma_5/BiasAdd/ReadVariableOpReadVariableOp(chroma_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
chroma_5/BiasAddBiasAddchroma_5/MatMul:product:0'chroma_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
chroma_5/ReluReluchroma_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
chroma_6/MatMul/ReadVariableOpReadVariableOp'chroma_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
chroma_6/MatMulMatMulchroma_5/Relu:activations:0&chroma_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
chroma_6/BiasAdd/ReadVariableOpReadVariableOp(chroma_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
chroma_6/BiasAddBiasAddchroma_6/MatMul:product:0'chroma_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
chroma_6/ReluReluchroma_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
chroma_7/MatMul/ReadVariableOpReadVariableOp'chroma_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
chroma_7/MatMulMatMulchroma_6/Relu:activations:0&chroma_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
chroma_7/BiasAdd/ReadVariableOpReadVariableOp(chroma_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
chroma_7/BiasAddBiasAddchroma_7/MatMul:product:0'chroma_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
chroma_7/ReluReluchroma_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#chroma_output/MatMul/ReadVariableOpReadVariableOp,chroma_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
chroma_output/MatMulMatMulchroma_7/Relu:activations:0+chroma_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$chroma_output/BiasAdd/ReadVariableOpReadVariableOp-chroma_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
chroma_output/BiasAddBiasAddchroma_output/MatMul:product:0,chroma_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
chroma_output/ReluReluchroma_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
IdentityIdentity chroma_output/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp ^chroma_1/BiasAdd/ReadVariableOp^chroma_1/MatMul/ReadVariableOp ^chroma_2/BiasAdd/ReadVariableOp^chroma_2/MatMul/ReadVariableOp ^chroma_3/BiasAdd/ReadVariableOp^chroma_3/MatMul/ReadVariableOp ^chroma_4/BiasAdd/ReadVariableOp^chroma_4/MatMul/ReadVariableOp ^chroma_5/BiasAdd/ReadVariableOp^chroma_5/MatMul/ReadVariableOp ^chroma_6/BiasAdd/ReadVariableOp^chroma_6/MatMul/ReadVariableOp ^chroma_7/BiasAdd/ReadVariableOp^chroma_7/MatMul/ReadVariableOp%^chroma_output/BiasAdd/ReadVariableOp$^chroma_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2B
chroma_1/BiasAdd/ReadVariableOpchroma_1/BiasAdd/ReadVariableOp2@
chroma_1/MatMul/ReadVariableOpchroma_1/MatMul/ReadVariableOp2B
chroma_2/BiasAdd/ReadVariableOpchroma_2/BiasAdd/ReadVariableOp2@
chroma_2/MatMul/ReadVariableOpchroma_2/MatMul/ReadVariableOp2B
chroma_3/BiasAdd/ReadVariableOpchroma_3/BiasAdd/ReadVariableOp2@
chroma_3/MatMul/ReadVariableOpchroma_3/MatMul/ReadVariableOp2B
chroma_4/BiasAdd/ReadVariableOpchroma_4/BiasAdd/ReadVariableOp2@
chroma_4/MatMul/ReadVariableOpchroma_4/MatMul/ReadVariableOp2B
chroma_5/BiasAdd/ReadVariableOpchroma_5/BiasAdd/ReadVariableOp2@
chroma_5/MatMul/ReadVariableOpchroma_5/MatMul/ReadVariableOp2B
chroma_6/BiasAdd/ReadVariableOpchroma_6/BiasAdd/ReadVariableOp2@
chroma_6/MatMul/ReadVariableOpchroma_6/MatMul/ReadVariableOp2B
chroma_7/BiasAdd/ReadVariableOpchroma_7/BiasAdd/ReadVariableOp2@
chroma_7/MatMul/ReadVariableOpchroma_7/MatMul/ReadVariableOp2L
$chroma_output/BiasAdd/ReadVariableOp$chroma_output/BiasAdd/ReadVariableOp2J
#chroma_output/MatMul/ReadVariableOp#chroma_output/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
­
,__inference_sequential_layer_call_fn_4630038
chroma_1_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallchroma_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4630003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namechroma_1_input
Å
­
,__inference_sequential_layer_call_fn_4630273
chroma_1_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallchroma_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4630201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namechroma_1_input
 

÷
E__inference_chroma_5_layer_call_and_return_conditional_losses_4629945

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_chroma_7_layer_call_and_return_conditional_losses_4630740

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÐF
·
G__inference_sequential_layer_call_and_return_conditional_losses_4630561

inputs;
'chroma_1_matmul_readvariableop_resource:
7
(chroma_1_biasadd_readvariableop_resource:	;
'chroma_2_matmul_readvariableop_resource:
7
(chroma_2_biasadd_readvariableop_resource:	;
'chroma_3_matmul_readvariableop_resource:
7
(chroma_3_biasadd_readvariableop_resource:	;
'chroma_4_matmul_readvariableop_resource:
7
(chroma_4_biasadd_readvariableop_resource:	:
'chroma_5_matmul_readvariableop_resource:	@6
(chroma_5_biasadd_readvariableop_resource:@9
'chroma_6_matmul_readvariableop_resource:@ 6
(chroma_6_biasadd_readvariableop_resource: 9
'chroma_7_matmul_readvariableop_resource:  6
(chroma_7_biasadd_readvariableop_resource: >
,chroma_output_matmul_readvariableop_resource: ;
-chroma_output_biasadd_readvariableop_resource:
identity¢chroma_1/BiasAdd/ReadVariableOp¢chroma_1/MatMul/ReadVariableOp¢chroma_2/BiasAdd/ReadVariableOp¢chroma_2/MatMul/ReadVariableOp¢chroma_3/BiasAdd/ReadVariableOp¢chroma_3/MatMul/ReadVariableOp¢chroma_4/BiasAdd/ReadVariableOp¢chroma_4/MatMul/ReadVariableOp¢chroma_5/BiasAdd/ReadVariableOp¢chroma_5/MatMul/ReadVariableOp¢chroma_6/BiasAdd/ReadVariableOp¢chroma_6/MatMul/ReadVariableOp¢chroma_7/BiasAdd/ReadVariableOp¢chroma_7/MatMul/ReadVariableOp¢$chroma_output/BiasAdd/ReadVariableOp¢#chroma_output/MatMul/ReadVariableOp
chroma_1/MatMul/ReadVariableOpReadVariableOp'chroma_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0|
chroma_1/MatMulMatMulinputs&chroma_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_1/BiasAdd/ReadVariableOpReadVariableOp(chroma_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_1/BiasAddBiasAddchroma_1/MatMul:product:0'chroma_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_1/ReluReluchroma_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_2/MatMul/ReadVariableOpReadVariableOp'chroma_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
chroma_2/MatMulMatMulchroma_1/Relu:activations:0&chroma_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_2/BiasAdd/ReadVariableOpReadVariableOp(chroma_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_2/BiasAddBiasAddchroma_2/MatMul:product:0'chroma_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_2/ReluReluchroma_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_3/MatMul/ReadVariableOpReadVariableOp'chroma_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
chroma_3/MatMulMatMulchroma_2/Relu:activations:0&chroma_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_3/BiasAdd/ReadVariableOpReadVariableOp(chroma_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_3/BiasAddBiasAddchroma_3/MatMul:product:0'chroma_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_3/ReluReluchroma_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_4/MatMul/ReadVariableOpReadVariableOp'chroma_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
chroma_4/MatMulMatMulchroma_3/Relu:activations:0&chroma_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_4/BiasAdd/ReadVariableOpReadVariableOp(chroma_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
chroma_4/BiasAddBiasAddchroma_4/MatMul:product:0'chroma_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
chroma_4/ReluReluchroma_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
chroma_5/MatMul/ReadVariableOpReadVariableOp'chroma_5_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
chroma_5/MatMulMatMulchroma_4/Relu:activations:0&chroma_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
chroma_5/BiasAdd/ReadVariableOpReadVariableOp(chroma_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
chroma_5/BiasAddBiasAddchroma_5/MatMul:product:0'chroma_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
chroma_5/ReluReluchroma_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
chroma_6/MatMul/ReadVariableOpReadVariableOp'chroma_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
chroma_6/MatMulMatMulchroma_5/Relu:activations:0&chroma_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
chroma_6/BiasAdd/ReadVariableOpReadVariableOp(chroma_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
chroma_6/BiasAddBiasAddchroma_6/MatMul:product:0'chroma_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
chroma_6/ReluReluchroma_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
chroma_7/MatMul/ReadVariableOpReadVariableOp'chroma_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
chroma_7/MatMulMatMulchroma_6/Relu:activations:0&chroma_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
chroma_7/BiasAdd/ReadVariableOpReadVariableOp(chroma_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
chroma_7/BiasAddBiasAddchroma_7/MatMul:product:0'chroma_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
chroma_7/ReluReluchroma_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#chroma_output/MatMul/ReadVariableOpReadVariableOp,chroma_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
chroma_output/MatMulMatMulchroma_7/Relu:activations:0+chroma_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$chroma_output/BiasAdd/ReadVariableOpReadVariableOp-chroma_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
chroma_output/BiasAddBiasAddchroma_output/MatMul:product:0,chroma_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
chroma_output/ReluReluchroma_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
IdentityIdentity chroma_output/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp ^chroma_1/BiasAdd/ReadVariableOp^chroma_1/MatMul/ReadVariableOp ^chroma_2/BiasAdd/ReadVariableOp^chroma_2/MatMul/ReadVariableOp ^chroma_3/BiasAdd/ReadVariableOp^chroma_3/MatMul/ReadVariableOp ^chroma_4/BiasAdd/ReadVariableOp^chroma_4/MatMul/ReadVariableOp ^chroma_5/BiasAdd/ReadVariableOp^chroma_5/MatMul/ReadVariableOp ^chroma_6/BiasAdd/ReadVariableOp^chroma_6/MatMul/ReadVariableOp ^chroma_7/BiasAdd/ReadVariableOp^chroma_7/MatMul/ReadVariableOp%^chroma_output/BiasAdd/ReadVariableOp$^chroma_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2B
chroma_1/BiasAdd/ReadVariableOpchroma_1/BiasAdd/ReadVariableOp2@
chroma_1/MatMul/ReadVariableOpchroma_1/MatMul/ReadVariableOp2B
chroma_2/BiasAdd/ReadVariableOpchroma_2/BiasAdd/ReadVariableOp2@
chroma_2/MatMul/ReadVariableOpchroma_2/MatMul/ReadVariableOp2B
chroma_3/BiasAdd/ReadVariableOpchroma_3/BiasAdd/ReadVariableOp2@
chroma_3/MatMul/ReadVariableOpchroma_3/MatMul/ReadVariableOp2B
chroma_4/BiasAdd/ReadVariableOpchroma_4/BiasAdd/ReadVariableOp2@
chroma_4/MatMul/ReadVariableOpchroma_4/MatMul/ReadVariableOp2B
chroma_5/BiasAdd/ReadVariableOpchroma_5/BiasAdd/ReadVariableOp2@
chroma_5/MatMul/ReadVariableOpchroma_5/MatMul/ReadVariableOp2B
chroma_6/BiasAdd/ReadVariableOpchroma_6/BiasAdd/ReadVariableOp2@
chroma_6/MatMul/ReadVariableOpchroma_6/MatMul/ReadVariableOp2B
chroma_7/BiasAdd/ReadVariableOpchroma_7/BiasAdd/ReadVariableOp2@
chroma_7/MatMul/ReadVariableOpchroma_7/MatMul/ReadVariableOp2L
$chroma_output/BiasAdd/ReadVariableOp$chroma_output/BiasAdd/ReadVariableOp2J
#chroma_output/MatMul/ReadVariableOp#chroma_output/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

/__inference_chroma_output_layer_call_fn_4630749

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_chroma_output_layer_call_and_return_conditional_losses_4629996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
+
Â
G__inference_sequential_layer_call_and_return_conditional_losses_4630361
chroma_1_input$
chroma_1_4630320:

chroma_1_4630322:	$
chroma_2_4630325:

chroma_2_4630327:	$
chroma_3_4630330:

chroma_3_4630332:	$
chroma_4_4630335:

chroma_4_4630337:	#
chroma_5_4630340:	@
chroma_5_4630342:@"
chroma_6_4630345:@ 
chroma_6_4630347: "
chroma_7_4630350:  
chroma_7_4630352: '
chroma_output_4630355: #
chroma_output_4630357:
identity¢ chroma_1/StatefulPartitionedCall¢ chroma_2/StatefulPartitionedCall¢ chroma_3/StatefulPartitionedCall¢ chroma_4/StatefulPartitionedCall¢ chroma_5/StatefulPartitionedCall¢ chroma_6/StatefulPartitionedCall¢ chroma_7/StatefulPartitionedCall¢%chroma_output/StatefulPartitionedCallü
 chroma_1/StatefulPartitionedCallStatefulPartitionedCallchroma_1_inputchroma_1_4630320chroma_1_4630322*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_1_layer_call_and_return_conditional_losses_4629877
 chroma_2/StatefulPartitionedCallStatefulPartitionedCall)chroma_1/StatefulPartitionedCall:output:0chroma_2_4630325chroma_2_4630327*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_2_layer_call_and_return_conditional_losses_4629894
 chroma_3/StatefulPartitionedCallStatefulPartitionedCall)chroma_2/StatefulPartitionedCall:output:0chroma_3_4630330chroma_3_4630332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_3_layer_call_and_return_conditional_losses_4629911
 chroma_4/StatefulPartitionedCallStatefulPartitionedCall)chroma_3/StatefulPartitionedCall:output:0chroma_4_4630335chroma_4_4630337*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_4_layer_call_and_return_conditional_losses_4629928
 chroma_5/StatefulPartitionedCallStatefulPartitionedCall)chroma_4/StatefulPartitionedCall:output:0chroma_5_4630340chroma_5_4630342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_5_layer_call_and_return_conditional_losses_4629945
 chroma_6/StatefulPartitionedCallStatefulPartitionedCall)chroma_5/StatefulPartitionedCall:output:0chroma_6_4630345chroma_6_4630347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_6_layer_call_and_return_conditional_losses_4629962
 chroma_7/StatefulPartitionedCallStatefulPartitionedCall)chroma_6/StatefulPartitionedCall:output:0chroma_7_4630350chroma_7_4630352*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_7_layer_call_and_return_conditional_losses_4629979ª
%chroma_output/StatefulPartitionedCallStatefulPartitionedCall)chroma_7/StatefulPartitionedCall:output:0chroma_output_4630355chroma_output_4630357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_chroma_output_layer_call_and_return_conditional_losses_4629996}
IdentityIdentity.chroma_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp!^chroma_1/StatefulPartitionedCall!^chroma_2/StatefulPartitionedCall!^chroma_3/StatefulPartitionedCall!^chroma_4/StatefulPartitionedCall!^chroma_5/StatefulPartitionedCall!^chroma_6/StatefulPartitionedCall!^chroma_7/StatefulPartitionedCall&^chroma_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 chroma_1/StatefulPartitionedCall chroma_1/StatefulPartitionedCall2D
 chroma_2/StatefulPartitionedCall chroma_2/StatefulPartitionedCall2D
 chroma_3/StatefulPartitionedCall chroma_3/StatefulPartitionedCall2D
 chroma_4/StatefulPartitionedCall chroma_4/StatefulPartitionedCall2D
 chroma_5/StatefulPartitionedCall chroma_5/StatefulPartitionedCall2D
 chroma_6/StatefulPartitionedCall chroma_6/StatefulPartitionedCall2D
 chroma_7/StatefulPartitionedCall chroma_7/StatefulPartitionedCall2N
%chroma_output/StatefulPartitionedCall%chroma_output/StatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namechroma_1_input
Ë

*__inference_chroma_1_layer_call_fn_4630609

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_1_layer_call_and_return_conditional_losses_4629877p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

*__inference_chroma_5_layer_call_fn_4630689

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_5_layer_call_and_return_conditional_losses_4629945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_2_layer_call_and_return_conditional_losses_4629894

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

÷
E__inference_chroma_5_layer_call_and_return_conditional_losses_4630700

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_1_layer_call_and_return_conditional_losses_4629877

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_1_layer_call_and_return_conditional_losses_4630620

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
¥
,__inference_sequential_layer_call_fn_4630441

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4630201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

*__inference_chroma_3_layer_call_fn_4630649

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_3_layer_call_and_return_conditional_losses_4629911p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
¥
,__inference_sequential_layer_call_fn_4630404

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4630003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý*
º
G__inference_sequential_layer_call_and_return_conditional_losses_4630201

inputs$
chroma_1_4630160:

chroma_1_4630162:	$
chroma_2_4630165:

chroma_2_4630167:	$
chroma_3_4630170:

chroma_3_4630172:	$
chroma_4_4630175:

chroma_4_4630177:	#
chroma_5_4630180:	@
chroma_5_4630182:@"
chroma_6_4630185:@ 
chroma_6_4630187: "
chroma_7_4630190:  
chroma_7_4630192: '
chroma_output_4630195: #
chroma_output_4630197:
identity¢ chroma_1/StatefulPartitionedCall¢ chroma_2/StatefulPartitionedCall¢ chroma_3/StatefulPartitionedCall¢ chroma_4/StatefulPartitionedCall¢ chroma_5/StatefulPartitionedCall¢ chroma_6/StatefulPartitionedCall¢ chroma_7/StatefulPartitionedCall¢%chroma_output/StatefulPartitionedCallô
 chroma_1/StatefulPartitionedCallStatefulPartitionedCallinputschroma_1_4630160chroma_1_4630162*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_1_layer_call_and_return_conditional_losses_4629877
 chroma_2/StatefulPartitionedCallStatefulPartitionedCall)chroma_1/StatefulPartitionedCall:output:0chroma_2_4630165chroma_2_4630167*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_2_layer_call_and_return_conditional_losses_4629894
 chroma_3/StatefulPartitionedCallStatefulPartitionedCall)chroma_2/StatefulPartitionedCall:output:0chroma_3_4630170chroma_3_4630172*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_3_layer_call_and_return_conditional_losses_4629911
 chroma_4/StatefulPartitionedCallStatefulPartitionedCall)chroma_3/StatefulPartitionedCall:output:0chroma_4_4630175chroma_4_4630177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_4_layer_call_and_return_conditional_losses_4629928
 chroma_5/StatefulPartitionedCallStatefulPartitionedCall)chroma_4/StatefulPartitionedCall:output:0chroma_5_4630180chroma_5_4630182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_5_layer_call_and_return_conditional_losses_4629945
 chroma_6/StatefulPartitionedCallStatefulPartitionedCall)chroma_5/StatefulPartitionedCall:output:0chroma_6_4630185chroma_6_4630187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_6_layer_call_and_return_conditional_losses_4629962
 chroma_7/StatefulPartitionedCallStatefulPartitionedCall)chroma_6/StatefulPartitionedCall:output:0chroma_7_4630190chroma_7_4630192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_7_layer_call_and_return_conditional_losses_4629979ª
%chroma_output/StatefulPartitionedCallStatefulPartitionedCall)chroma_7/StatefulPartitionedCall:output:0chroma_output_4630195chroma_output_4630197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_chroma_output_layer_call_and_return_conditional_losses_4629996}
IdentityIdentity.chroma_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp!^chroma_1/StatefulPartitionedCall!^chroma_2/StatefulPartitionedCall!^chroma_3/StatefulPartitionedCall!^chroma_4/StatefulPartitionedCall!^chroma_5/StatefulPartitionedCall!^chroma_6/StatefulPartitionedCall!^chroma_7/StatefulPartitionedCall&^chroma_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 chroma_1/StatefulPartitionedCall chroma_1/StatefulPartitionedCall2D
 chroma_2/StatefulPartitionedCall chroma_2/StatefulPartitionedCall2D
 chroma_3/StatefulPartitionedCall chroma_3/StatefulPartitionedCall2D
 chroma_4/StatefulPartitionedCall chroma_4/StatefulPartitionedCall2D
 chroma_5/StatefulPartitionedCall chroma_5/StatefulPartitionedCall2D
 chroma_6/StatefulPartitionedCall chroma_6/StatefulPartitionedCall2D
 chroma_7/StatefulPartitionedCall chroma_7/StatefulPartitionedCall2N
%chroma_output/StatefulPartitionedCall%chroma_output/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_4_layer_call_and_return_conditional_losses_4630680

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_2_layer_call_and_return_conditional_losses_4630640

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
úr

 __inference__traced_save_4630960
file_prefix.
*savev2_chroma_1_kernel_read_readvariableop,
(savev2_chroma_1_bias_read_readvariableop.
*savev2_chroma_2_kernel_read_readvariableop,
(savev2_chroma_2_bias_read_readvariableop.
*savev2_chroma_3_kernel_read_readvariableop,
(savev2_chroma_3_bias_read_readvariableop.
*savev2_chroma_4_kernel_read_readvariableop,
(savev2_chroma_4_bias_read_readvariableop.
*savev2_chroma_5_kernel_read_readvariableop,
(savev2_chroma_5_bias_read_readvariableop.
*savev2_chroma_6_kernel_read_readvariableop,
(savev2_chroma_6_bias_read_readvariableop.
*savev2_chroma_7_kernel_read_readvariableop,
(savev2_chroma_7_bias_read_readvariableop3
/savev2_chroma_output_kernel_read_readvariableop1
-savev2_chroma_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_chroma_1_kernel_m_read_readvariableop3
/savev2_adam_chroma_1_bias_m_read_readvariableop5
1savev2_adam_chroma_2_kernel_m_read_readvariableop3
/savev2_adam_chroma_2_bias_m_read_readvariableop5
1savev2_adam_chroma_3_kernel_m_read_readvariableop3
/savev2_adam_chroma_3_bias_m_read_readvariableop5
1savev2_adam_chroma_4_kernel_m_read_readvariableop3
/savev2_adam_chroma_4_bias_m_read_readvariableop5
1savev2_adam_chroma_5_kernel_m_read_readvariableop3
/savev2_adam_chroma_5_bias_m_read_readvariableop5
1savev2_adam_chroma_6_kernel_m_read_readvariableop3
/savev2_adam_chroma_6_bias_m_read_readvariableop5
1savev2_adam_chroma_7_kernel_m_read_readvariableop3
/savev2_adam_chroma_7_bias_m_read_readvariableop:
6savev2_adam_chroma_output_kernel_m_read_readvariableop8
4savev2_adam_chroma_output_bias_m_read_readvariableop5
1savev2_adam_chroma_1_kernel_v_read_readvariableop3
/savev2_adam_chroma_1_bias_v_read_readvariableop5
1savev2_adam_chroma_2_kernel_v_read_readvariableop3
/savev2_adam_chroma_2_bias_v_read_readvariableop5
1savev2_adam_chroma_3_kernel_v_read_readvariableop3
/savev2_adam_chroma_3_bias_v_read_readvariableop5
1savev2_adam_chroma_4_kernel_v_read_readvariableop3
/savev2_adam_chroma_4_bias_v_read_readvariableop5
1savev2_adam_chroma_5_kernel_v_read_readvariableop3
/savev2_adam_chroma_5_bias_v_read_readvariableop5
1savev2_adam_chroma_6_kernel_v_read_readvariableop3
/savev2_adam_chroma_6_bias_v_read_readvariableop5
1savev2_adam_chroma_7_kernel_v_read_readvariableop3
/savev2_adam_chroma_7_bias_v_read_readvariableop:
6savev2_adam_chroma_output_kernel_v_read_readvariableop8
4savev2_adam_chroma_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: !
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*¬ 
value¢ B <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHè
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_chroma_1_kernel_read_readvariableop(savev2_chroma_1_bias_read_readvariableop*savev2_chroma_2_kernel_read_readvariableop(savev2_chroma_2_bias_read_readvariableop*savev2_chroma_3_kernel_read_readvariableop(savev2_chroma_3_bias_read_readvariableop*savev2_chroma_4_kernel_read_readvariableop(savev2_chroma_4_bias_read_readvariableop*savev2_chroma_5_kernel_read_readvariableop(savev2_chroma_5_bias_read_readvariableop*savev2_chroma_6_kernel_read_readvariableop(savev2_chroma_6_bias_read_readvariableop*savev2_chroma_7_kernel_read_readvariableop(savev2_chroma_7_bias_read_readvariableop/savev2_chroma_output_kernel_read_readvariableop-savev2_chroma_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_chroma_1_kernel_m_read_readvariableop/savev2_adam_chroma_1_bias_m_read_readvariableop1savev2_adam_chroma_2_kernel_m_read_readvariableop/savev2_adam_chroma_2_bias_m_read_readvariableop1savev2_adam_chroma_3_kernel_m_read_readvariableop/savev2_adam_chroma_3_bias_m_read_readvariableop1savev2_adam_chroma_4_kernel_m_read_readvariableop/savev2_adam_chroma_4_bias_m_read_readvariableop1savev2_adam_chroma_5_kernel_m_read_readvariableop/savev2_adam_chroma_5_bias_m_read_readvariableop1savev2_adam_chroma_6_kernel_m_read_readvariableop/savev2_adam_chroma_6_bias_m_read_readvariableop1savev2_adam_chroma_7_kernel_m_read_readvariableop/savev2_adam_chroma_7_bias_m_read_readvariableop6savev2_adam_chroma_output_kernel_m_read_readvariableop4savev2_adam_chroma_output_bias_m_read_readvariableop1savev2_adam_chroma_1_kernel_v_read_readvariableop/savev2_adam_chroma_1_bias_v_read_readvariableop1savev2_adam_chroma_2_kernel_v_read_readvariableop/savev2_adam_chroma_2_bias_v_read_readvariableop1savev2_adam_chroma_3_kernel_v_read_readvariableop/savev2_adam_chroma_3_bias_v_read_readvariableop1savev2_adam_chroma_4_kernel_v_read_readvariableop/savev2_adam_chroma_4_bias_v_read_readvariableop1savev2_adam_chroma_5_kernel_v_read_readvariableop/savev2_adam_chroma_5_bias_v_read_readvariableop1savev2_adam_chroma_6_kernel_v_read_readvariableop/savev2_adam_chroma_6_bias_v_read_readvariableop1savev2_adam_chroma_7_kernel_v_read_readvariableop/savev2_adam_chroma_7_bias_v_read_readvariableop6savev2_adam_chroma_output_kernel_v_read_readvariableop4savev2_adam_chroma_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ö
_input_shapesÄ
Á: :
::
::
::
::	@:@:@ : :  : : :: : : : : : : : : : : :
::
::
::
::	@:@:@ : :  : : ::
::
::
::
::	@:@:@ : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%	!

_output_shapes
:	@: 


_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::%$!

_output_shapes
:	@: %

_output_shapes
:@:$& 

_output_shapes

:@ : '

_output_shapes
: :$( 

_output_shapes

:  : )

_output_shapes
: :$* 

_output_shapes

: : +

_output_shapes
::&,"
 
_output_shapes
:
:!-

_output_shapes	
::&."
 
_output_shapes
:
:!/

_output_shapes	
::&0"
 
_output_shapes
:
:!1

_output_shapes	
::&2"
 
_output_shapes
:
:!3

_output_shapes	
::%4!

_output_shapes
:	@: 5

_output_shapes
:@:$6 

_output_shapes

:@ : 7

_output_shapes
: :$8 

_output_shapes

:  : 9

_output_shapes
: :$: 

_output_shapes

: : ;

_output_shapes
::<

_output_shapes
: 


ö
E__inference_chroma_7_layer_call_and_return_conditional_losses_4629979

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä

*__inference_chroma_6_layer_call_fn_4630709

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_6_layer_call_and_return_conditional_losses_4629962o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë

*__inference_chroma_4_layer_call_fn_4630669

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_4_layer_call_and_return_conditional_losses_4629928p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_chroma_3_layer_call_and_return_conditional_losses_4629911

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

*__inference_chroma_2_layer_call_fn_4630629

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_chroma_2_layer_call_and_return_conditional_losses_4629894p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
J
chroma_1_input8
 serving_default_chroma_1_input:0ÿÿÿÿÿÿÿÿÿA
chroma_output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ô

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
»

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
»

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
»

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
»

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer

Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemmmm"m#m*m+m2m3m:m ;m¡Bm¢Cm£Jm¤Km¥v¦v§v¨v©"vª#v«*v¬+v­2v®3v¯:v°;v±Bv²Cv³Jv´Kvµ"
	optimizer

0
1
2
3
"4
#5
*6
+7
28
39
:10
;11
B12
C13
J14
K15"
trackable_list_wrapper

0
1
2
3
"4
#5
*6
+7
28
39
:10
;11
B12
C13
J14
K15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_layer_call_fn_4630038
,__inference_sequential_layer_call_fn_4630404
,__inference_sequential_layer_call_fn_4630441
,__inference_sequential_layer_call_fn_4630273À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_4630501
G__inference_sequential_layer_call_and_return_conditional_losses_4630561
G__inference_sequential_layer_call_and_return_conditional_losses_4630317
G__inference_sequential_layer_call_and_return_conditional_losses_4630361À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÔBÑ
"__inference__wrapped_model_4629859chroma_1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
\serving_default"
signature_map
#:!
2chroma_1/kernel
:2chroma_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_1_layer_call_fn_4630609¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_1_layer_call_and_return_conditional_losses_4630620¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
2chroma_2/kernel
:2chroma_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_2_layer_call_fn_4630629¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_2_layer_call_and_return_conditional_losses_4630640¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
2chroma_3/kernel
:2chroma_3/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_3_layer_call_fn_4630649¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_3_layer_call_and_return_conditional_losses_4630660¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
2chroma_4/kernel
:2chroma_4/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_4_layer_call_fn_4630669¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_4_layer_call_and_return_conditional_losses_4630680¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	@2chroma_5/kernel
:@2chroma_5/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_5_layer_call_fn_4630689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_5_layer_call_and_return_conditional_losses_4630700¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:@ 2chroma_6/kernel
: 2chroma_6/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_6_layer_call_fn_4630709¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_6_layer_call_and_return_conditional_losses_4630720¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:  2chroma_7/kernel
: 2chroma_7/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_chroma_7_layer_call_fn_4630729¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_chroma_7_layer_call_and_return_conditional_losses_4630740¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$ 2chroma_output/kernel
 :2chroma_output/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_chroma_output_layer_call_fn_4630749¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_chroma_output_layer_call_and_return_conditional_losses_4630760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÓBÐ
%__inference_signature_wrapper_4630600chroma_1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
(:&
2Adam/chroma_1/kernel/m
!:2Adam/chroma_1/bias/m
(:&
2Adam/chroma_2/kernel/m
!:2Adam/chroma_2/bias/m
(:&
2Adam/chroma_3/kernel/m
!:2Adam/chroma_3/bias/m
(:&
2Adam/chroma_4/kernel/m
!:2Adam/chroma_4/bias/m
':%	@2Adam/chroma_5/kernel/m
 :@2Adam/chroma_5/bias/m
&:$@ 2Adam/chroma_6/kernel/m
 : 2Adam/chroma_6/bias/m
&:$  2Adam/chroma_7/kernel/m
 : 2Adam/chroma_7/bias/m
+:) 2Adam/chroma_output/kernel/m
%:#2Adam/chroma_output/bias/m
(:&
2Adam/chroma_1/kernel/v
!:2Adam/chroma_1/bias/v
(:&
2Adam/chroma_2/kernel/v
!:2Adam/chroma_2/bias/v
(:&
2Adam/chroma_3/kernel/v
!:2Adam/chroma_3/bias/v
(:&
2Adam/chroma_4/kernel/v
!:2Adam/chroma_4/bias/v
':%	@2Adam/chroma_5/kernel/v
 :@2Adam/chroma_5/bias/v
&:$@ 2Adam/chroma_6/kernel/v
 : 2Adam/chroma_6/bias/v
&:$  2Adam/chroma_7/kernel/v
 : 2Adam/chroma_7/bias/v
+:) 2Adam/chroma_output/kernel/v
%:#2Adam/chroma_output/bias/v²
"__inference__wrapped_model_4629859"#*+23:;BCJK8¢5
.¢+
)&
chroma_1_inputÿÿÿÿÿÿÿÿÿ
ª "=ª:
8
chroma_output'$
chroma_outputÿÿÿÿÿÿÿÿÿ§
E__inference_chroma_1_layer_call_and_return_conditional_losses_4630620^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_chroma_1_layer_call_fn_4630609Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_chroma_2_layer_call_and_return_conditional_losses_4630640^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_chroma_2_layer_call_fn_4630629Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_chroma_3_layer_call_and_return_conditional_losses_4630660^"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_chroma_3_layer_call_fn_4630649Q"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_chroma_4_layer_call_and_return_conditional_losses_4630680^*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_chroma_4_layer_call_fn_4630669Q*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_chroma_5_layer_call_and_return_conditional_losses_4630700]230¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
*__inference_chroma_5_layer_call_fn_4630689P230¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¥
E__inference_chroma_6_layer_call_and_return_conditional_losses_4630720\:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_chroma_6_layer_call_fn_4630709O:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_chroma_7_layer_call_and_return_conditional_losses_4630740\BC/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_chroma_7_layer_call_fn_4630729OBC/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ª
J__inference_chroma_output_layer_call_and_return_conditional_losses_4630760\JK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_chroma_output_layer_call_fn_4630749OJK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÆ
G__inference_sequential_layer_call_and_return_conditional_losses_4630317{"#*+23:;BCJK@¢=
6¢3
)&
chroma_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
G__inference_sequential_layer_call_and_return_conditional_losses_4630361{"#*+23:;BCJK@¢=
6¢3
)&
chroma_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
G__inference_sequential_layer_call_and_return_conditional_losses_4630501s"#*+23:;BCJK8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
G__inference_sequential_layer_call_and_return_conditional_losses_4630561s"#*+23:;BCJK8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_layer_call_fn_4630038n"#*+23:;BCJK@¢=
6¢3
)&
chroma_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_4630273n"#*+23:;BCJK@¢=
6¢3
)&
chroma_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_4630404f"#*+23:;BCJK8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_4630441f"#*+23:;BCJK8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÇ
%__inference_signature_wrapper_4630600"#*+23:;BCJKJ¢G
¢ 
@ª=
;
chroma_1_input)&
chroma_1_inputÿÿÿÿÿÿÿÿÿ"=ª:
8
chroma_output'$
chroma_outputÿÿÿÿÿÿÿÿÿ