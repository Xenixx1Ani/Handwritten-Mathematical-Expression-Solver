# AI-ML Hackathon - No. Label Recognition

### Pranay Shah, Vinamr Jain, Anikool Saini

### July 2021

## 1 Introduction

We’ll try to recognise certain patterns among numerical expressions which
might help us recognise and classify the numbers provided in the data-set
(upon splitting the image into three parts) so that we can train the AI for
further computation.
Note-We must abstain from using other no.’s for the classification as that
will reduce the accuracy since the no’s themselves have been classified using
an imperfect operator-classifier AI. However, we need to come up with as
many permutation possible in order to increase the size of our training data-
set.I’ve tried my best to use only 0, 1 and 9 to classify other no.’s
in order to maximise the accuracy

## 2 Classification of zero

1. -9 is only obtainable for the expression 0 - 9. (first digit always 0)
2. 0 with addition operator only possible with 0 + 0.
3. 0 with division operator only possible for 0÷k, k∈[0,9]∩z

## 3 Classification of nine

1. -9 is only obtainable for the expression 0 - 9. (second digit always 9)
2. 81 can only be obtained for 9×9.
3. 9 with division operator only possible for 9÷ 1


## 4 A few generalisations

- check both digits first and see if any of them is 0 or 9 or 1. if so then
    once we know the operator it is quite simple to conclude the second no.
    for example-if 0 is present and operator is sum then k implies the other
    no. is k. set the rules for classifying other no.’s accordingly.note-for
    one we will be using the above rule for 0 and 9 only.

```
Examples-
```
1. - operator and first no. is 0 then -k implies second no. is k
    k∈[1,9]∩z
2. ÷operator and second digit is one then k implies first no. is k.
- as for the special cases pertaining to each no. I would be mentioning
them explicitly (given emphasis).
- another generalisation is for the square of the no.’s, although I have
mentioned them anyways.

## 5 Classification of one

1. 1 with multiplication operator only possible with 1 × 1
2. 1 with addition operator only possible with 1 + 0 , 0 + 1. (once
    you’ve recognised 0)
3. 2 with addition operator and if no zeroes present then only possible for
    1 + 1.(once you’ve recognised 0)
4. -8 can be obtained for 0 - 8 , 1 - 9.(once you’ve recognised 0 and
    9)
5. -1 and the first no. is 0 means 0 - 1.(once you’ve recognised 0)

## 6 Classification of two

1. 4 can only be obtained from 2 × 2 , 4 × 1 , 1 × 4. so if both of the no.’s
    are not one then 2 × 2 .(once you’ve recognised 1)


2. 2 with multiplication operator only possible for 2× 1 or 1 ×2.(once
    you’ve recognised 1)
3. 2 with addition operator with no ones present only possible for 2 + 0
    or 0 + 2.(once you’ve recognised 0 and 1)
4. -7 can be obtained for 2 - 9, 1 - 8, 0 - 7.(once you’ve recognised 0,
    and 9)
5. 2 with division operator and 1 as divisor implies 2÷1.(once you’ve
    recognised 1)
6. -2 and the first no. is 0 means 0 - 1.(once you’ve recognised 0)

## 7 Classification of three

1. 9 can only be obtained from 3 × 3 , 9 × 1 , 1 × 9. so if nine and one are
    not present implies 3 × 3 .(once you’ve recognised 1 and 9)
2. 3 with multiplication operator only possible for 3× 1 or 1 ×3.(once
    you’ve recognised 1)
3. 3 with division operator and 1 as divisor implies 3÷1.(once you’ve
    recognised 1)
4. -3 and the first no. is 0 means 0 - 3.(once you’ve recognised 0)

## 8 Classification of four

1. 4 with division operator and 1 as divisor implies 4÷1.(once you’ve
    recognised 1)
2. -4 and the first no. is 0 means 0 - 4.(once you’ve recognised 0)
3. 4 can only be obtained from 2× 2 , 4 × 1 , 1 ×4. so if one of the no. is
    one implies the other is 4.(once you’ve recognised 1)
4. -5 and second no. is 9 means first is 4 i.e 4-9.(once you’ve recognised
    9)


## 9 Classification of five

1. 25 can only be obtained from 5 × 5.

## 10 Classification of six

1. 36 can only be obtained from 4 × 9 , 9 × 4 , 6 × 6. so if nine is not
    present implies 6 × 6 .(once you’ve recognised 9)

## 11 Classification of seven

1. 49 can only be obtained from 7 × 7.

## 12 Classification of eight

1. 64 can only be obtained from 8 × 8.


