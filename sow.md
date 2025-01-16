# Exhibit A — Statement of work

*Project name.* Staking & supply side microeconomics of the Swarm network
*Consultant.* Shtuka Research OÜ

**Project Summary.** This study will describe and evaluate economic aspects of the Swarm network and protocol by means of economic modelling, find economic vulnerabilities, and propose improvements.

The Swarm economy consists of the following components:

* A token, BZZ, used for "staking" and payments.
* An oracle that provides a demand-side control by dictating the price customers pay for storage.
* A "staking" system that provides several supply-side controls (elaborated below).
* A periodically executing clearing (smart contract) system that redistributes customer payments to storage providers.
* A peer-to-peer payments scheme for bandwidth usage.

This study will focus on supply-side controls and aggregate performance indicators.

**Previous work.** In a previous article, we have demonstrated weaknesses in the demand-side controls of the network. In informal discussions, some ideas to address this — for example, retiring the use of BZZ for payments and using a stablecoin, such as xDAI, instead — were floated.

**Goal.** Evaluate the supply-side controls of the Swarm storage market, addressing the following questions:

1. What are the objectives of the supply-side controls, and does the current system achieve them?
2. If not, what alternative models can be proposed to correct its deficiencies?
3. What metrics should we be monitoring to ensure the system continues to meet its targets?

**Method.** Produce a mathematical model of the Swarm economy that incorporates the details of supply side controls, i.e. the "staking" system. Evaluate this model against a list of economic objectives produced in collaboration with the client.

**Discussion.**
In its current form, Swarm nodes must "stake" (burn) BZZ tokens in exchange for a revocable authorization to participate in the Swarm storage market by providing  a fixed amount of storage and earning a share of the resulting revenue. Interestingly, the share of revenue thus earned is related to the amount of BZZ burned, not the amount of storage provided. 

Some of the main objectives of this system are as follows:
1. Establish a cost — to wit, revocation of the authorization — of falling out of consensus over which chunks should be hosted in a neighbourhood of address space.
2. Remove an incentive — increasing ones share of the revenue — to Sybil by running multiple nodes in a neighbourhood, deduplicating the underlying storage. Instead, suppliers may equally increase their share of revenue simply by burning additional BZZ, without providing any additional physical infrastructure.

While Objective $1$ is a fairly typical application of a blockchain based staking system, and provides some trust-reduced guarantees about the durability of data storage on Swarm, Objective $2$ has the unusual feature that suppliers may increase their share of revenue without making a greater contribution to the service provided by the network. Moreover, even with the current staking system there are still some incentives to Sybil, and it is inexpensive to do so. This audit will investigate some of the consequences of the revenue distribution weighting system.

Perhaps more seriously, the current system incentivises continuously topping up stake, erecting an ever-growing barrier to entry and hence a stagnation and centralization vector.

On the macro scale, the aggregate behaviour of nodes in the Swarm network has implications for the quality and revenue of the Swarm storage application. Hence, the Swarm maintainers will want to continuously monitor and respond to economic metrics that influence or contribute to forecasts for these outcomes. 
* Conditions and mechanism of convergence on an equilibrium replication rate and total revenue.
* Predictors of economic shocks through "hidden" factors such as supplier opex.

**Milestones.**

1. In discussion with Client, enumerate objectives for supply-side controls.
2. Show that as long as rewards are bounded below, at equilibrium total stake per neighbourhood increases without limit. Formulate the optimal staking problem on different time horizons.
3. Describe and evaluate alternative staking models on the same basis as in part 2:
   1. "Tax" system (like current system, but depositing stake only buys a *temporary* right to provide storage).
   2. Withdrawable stake system with yield.
   3. System where instead of being discretionary, stake is a function of macro conditions and the amount of storage provided.
4. Describe the process of macro convergence on equilibrium. Establish conditions under which the market clears at the target replication rate.
5. Identify aggregate economic indicators relevant for monitoring and forecasting Swarm storage performance.

**Deliverables.**

- Monthly update reports, prepared using MarkDown.
- A final report, prepared using MarkDown, containing a detailed writeup of all the milestones.
- SwIP[s], upon discussion with Client.
- Formal thoroughly referenced academic-style report, prepared to publication quality using LaTeX.
