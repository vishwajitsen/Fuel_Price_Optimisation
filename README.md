Fuel Price Optimization Using Genetic Algorithms – Summary Report

1. Understanding of the Problem
Fuel retailers operate in highly competitive and price-sensitive markets. Setting the optimal daily fuel price is critical to balance sales volume, market share, and profitability.
•	If prices are too high → demand drops, customers switch to competitors.
•	If prices are too low → demand increases but margins shrink, hurting profit.
The optimization challenge:
•	Identify the sweet spot price where profit is maximized, accounting for customer sensitivity to price and competitor reactions.
•	Incorporate historical data (sales, prices, competitor data, demand trends) and predictive modeling to simulate today’s demand at different possible price points.
A Genetic Algorithm (GA) was used to efficiently search the wide range of possible prices and converge on the most profitable solution.

2. Key Assumptions
Several assumptions were necessary to simplify and make the problem tractable:
1.	Demand Function Predictability
o	Sales volumes respond in a relatively stable, measurable way to price changes, captured by the trained predictive model.
2.	Stationarity of Market Conditions
o	Competitive effects and demand patterns from historical data are assumed to continue in the short term (near future).
3.	No External Shocks
o	Unexpected events (fuel shortages, government policy shifts, disasters) are not considered.
4.	Smooth Price Sensitivity
o	Demand decreases smoothly as prices rise and increases when prices drop, without sudden discontinuities.

5.	Profitability Calculation
o	Profit = (Price – Base Cost) × Predicted Sales Volume
o	Costs are assumed stable for the optimization period.
6.	Search Space Constraints
o	Prices are explored in fine increments (₹0.01) but bounded within a realistic band (e.g., ±₹10 around historical/market reference).

3. Chosen Methodology and Reasoning
Step 1: Predictive Modeling
•	A regression-based predictor (Random Forest) was trained on historical fuel sales data.
•	Input features: historical prices, competitor prices, day-of-week, seasonal demand, promotions, etc.
•	Output: Predicted demand (sales volume) for a given day and price.
Step 2: Profit Function
•	Profit was computed at different candidate price points 
Step 3: Optimization via Genetic Algorithm (GA)
GA was chosen because:
•	The relationship between price and profit is nonlinear (not a simple curve).
•	Classical optimization (gradient-based) may get stuck in local optima.
•	GA mimics natural selection to explore broadly, then refine toward optimal solutions.

4. Validation Results
Validation involved:
1.	Cross-validation of predictive model – Average R² used as weight (ensures GA trusts predictions more when the model is strong).
2.	Back testing – Applied the GA on historical days where actual sales and competitor prices are known.
o	Result: Optimized prices consistently yielded 5–10% higher profits than actual recorded prices.
3.	Robustness checks – Running GA multiple times with different seeds produced stable results, proving convergence reliability.

5. Example Output (using today_example. json)
Given today’s conditions:
•	Base Cost: 85.00 per liter
•	Competitor Prices: Around 95–96 per liter
•	Demand elasticity modeled from history
GA Result:
•	Recommended Price: 95.45
•	Expected Demand: ~14,381 liters
•	Expected Profit: ~1,39,212.34


6. Recommendations for Improvement:
To enhance the predictive power and robustness of the fuel price optimization model, additional features can be incorporated beyond the current dataset. These features would allow the model to better capture real-world dynamics and customer behavior, ultimately improving forecast accuracy and decision-making. Potential enhancements include:
	Weather conditions (rain, temperature, extreme events) affecting travel demand.
	Macroeconomic indicators such as inflation, crude oil prices, and currency exchange rates.
	Traffic and mobility data (vehicle density, commuting trends, fuel demand by time of day).
	Customer segmentation variables (fleet operators vs. individual consumers, loyalty membership).
	Seasonality and event markers (festivals, holidays, local events).
	Competitor pricing elasticity — how demand shifts when nearby stations change their prices.
	Promotions and marketing campaigns impacting short-term demand.
	Supply-side constraints such as inventory levels or refinery disruptions.

