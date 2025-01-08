+++
date = "2016-05-12T23:45:48+01:00"
title = "Project 7: FX-Trader"
+++

A derivatives trading system. This is my attempt at building a derivatives trading system.
In this first post I am going to outline the goals of the project and some of the early design decisions.

<!--more-->

This is an ambitious project so it will take a lot of posts.  

The ideals of this project are 

1. Volume: this solution is designed to handle large volumes of trades
2. Performance: this solution will be as fast as the hardware allows.
3. Extensibility: the solution will have in built extensibility.
4. Ease of use: the solution will be easy to use.
5. Pluggable: the system is designed to play nicely with other systems as part of a bigger solution.

For now I am going to focus on a very specific purpose and a very specific trade.

1. This project will load and display derivatives.
2. It will focus on just one derivative for now FX Forwards.

There is a reason I chose these two items, loading and displaying derivatives is difficult. 
You would not think it but it actually is, these is a very large amount of boiler plate stuff that has to be in place before
you can represent a derivative.

Why FX Forwards..... there are lots of reasons

1. They are popular, most large companies that have a branch in more than one country at some stage will need to trade
FX forward contracts.   
2. They are massive volume. In a typical trading environment you will see ten times as many of these guys as any other derivative.
The point here is that if you build a system that can handle these well then handling the other derivatives will be a matter of implementation.

## Technology Stack

For this project I am going to use [spring.io](https://spring.io "spring"). This takes care of a lot of the boiler plate stuff. 
Is very popular, so it will be familiar to a large group of developers.

