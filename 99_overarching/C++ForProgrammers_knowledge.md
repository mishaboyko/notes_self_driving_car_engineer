# C++ for Programmers
## Source
https://classroom.udacity.com/courses/ud999

## Set width in the console output
```c++
#include <iomanip>
std::cout << std::setw(10) << "word";
```

## Get input for std::in from file
### Compile

`g++ main.cpp -o main.out`

### Run program with input from file

// downside: space is end-of-line

// ./main.out stdin=open("input.txt", "r")

```c++
./main.out input.txt
```

// Solution: use getline() in code:
```
std::getline(std::cin, userName);
```


## Logical Operators

### Nice hack
```c++
    std::string TorF[] = {"False", "True"};
    
    //The && operator
    std::cout<<"A == C is "<<TorF[A==C];
    std::cout<<"\n(A ==C) && (B > D) is "<<TorF[(A ==C) && (B > D)];     
```
## User input type conversion

```c++
/*
* Option 1
*/
#include <sstream>

std::string userString;
int guess = -1;

std::cout<<"Guess a number between 0 and 100: ";
std::getline (std::cin,userString);
//convert to an int
std::stringstream(userString) >> guess;

/*
* Option 2
*/
std::cin>>givenInt;
std::cin>>givenFloat;
std::cin>>givenDouble;
//We need to use cin.ignore so cin will ignore the characters in the buffer leftover from the givenDouble
std::cin.ignore();
std::cin>>givenChar;
std::cout<<"character = "<<(char)givenChar<<"\n\n";
// Special case for printing addresses of the chars
std::cout<< "address character = " << (void *) &givenChar<<"\n\n";
std::cin.ignore();
std::getline (std::cin,givenString);
```



## Random numbers

```c++
#include <time.h> //added for the random number generator seed
#include <cstdlib>//added to use the rand function

    srand(time(NULL));  //set the seed for the random number generator
    target = rand() %100 + 1; //generate the 'random' number
```

## Pointers

```c++
 // this is an integer variable with value = 54
int a = 54; 

// this is a pointer that holds the address of the variable 'a'.
// if 'a' was a float, rather than int, so should be its pointer.
int * pointerToA = &a;  

// If we were to print pointerToA, we'd obtain the address of 'a':
std::cout << "pointerToA stores " << pointerToA << '\n';

// If we want to know what is stored in this address, we can dereference pointerToA:
std::cout << "pointerToA points to " << * pointerToA << '\n';
```

```c++
#include<iostream>
#include<string>

int main ()
{
  int * pointerI;
  int number;
  char character;
  char * pointerC;
  std::string sentence;
  std::string *pointerS;
  
  pointerI = &number;
  *pointerI = 45;
  
  pointerC = &character;
  *pointerC = 'f';
  
  pointerS = &sentence;
  *pointerS = "Hey look at me, I know pointers!";
  
  std::cout << "number = "<<number<<"\n";
  std::cout<<"character = "<<character<<"\n";
  std::cout<<"sentence = "<<sentence<<"\n";

  return 0;
}	
```



## C++ Functions

All C++ functions (except the special case of the main function) must have:

- A **declaration**: this is a statement of how the function is to be called.
- A **definition**: this is the statement(s) of the task the function performs when called



Pass by reference: 

```c++
void increment(int &input); //Note the addition of '&'
```

## Array as pointer

```c++
#include<iostream>
#include<iomanip>

//Pass the array as a pointer
void arrayAsPointer(int *array, int size);
//Pass the array as a sized array
void arraySized(int array[3], int size);
//Pass the array as an unsized array
void arrayUnSized(int array[], int size);

int main()
{
    const int size = 3;
    int array[size] = {33,66,99};
    //We are passing a pointer or reference to the array
    //so we will not know the size of the array
    //We have to pass the size to the function as well
    arrayAsPointer(array, size);
    arraySized(array, size);
    arrayUnSized(array, size);
    return 0;
}

void arrayAsPointer(int *array, int size)
{
    std::cout<<std::setw(5);
    for(int i=0; i<size; i++) 
        std::cout<<array[i]<<" ";
    std::cout<<"\n";
}

void arraySized(int array[3], int size)
{
    std::cout<<std::setw(5);
    for(int i=0; i<size; i++)
        std::cout<<array[i]<<" ";
    std::cout<<"\n";  
}

void arrayUnSized(int array[], int size)
{
    std::cout<<std::setw(5);
    for(int i=0; i<size; i++)
        std::cout<<array[i]<<" ";
    std::cout<<"\n";  
}

```

## Inheritance

```c++
#include<iostream>

using namespace std;
class A
{
     public: 
           //TODO: run the code with and without
           //the keyword 'virtual'
           /*virtual*/ void getMe();
           A();
};
void A::getMe()
{
    cout<<"A!";
}
A::A()
{
    cout<<" A constructor";
}
class B: public A
{
    public:
           void getMe();
           B();
};
B::B()
{
    cout<<" B constructor";
}
void B::getMe()
{
    cout<<"B!";
}

int main()
{
    cout<<"\nCreating instances:";
    A *instanceB;
    A *instanceA;
    cout<<"\nCreating 'new' instances:";
    cout<<"\n\tinstanceB: ";
    instanceB = new B();
    cout<<"\n\tinstanceA: ";
    instanceA = new A();
    
    cout<<"\ninstanceB->getMe() produces: ";
    instanceB->getMe();
    cout<<"\ninstanceA->getMe() produces: ";
    instanceA -> getMe();
    return 0;
}
```



```cpp
#include <type_traits>

template <class C, class P>

int IsDerivedFrom(C &c, P &p){
		return std::is_base_of<P, C>();
}

```

