
# INSERTION SORT (Portfolio Sorting)

def _calculate_portfolio_value(portfolio):
    stocks = portfolio.get("stocks", [])
    total = 0.0
    for s in stocks:
        price = s.get("purchase_price", s.get("price", 0))
        shares = s.get("shares", 1)
        total += price * shares
    return total


def insertion_sort_portfolios(portfolios, key="value"):
    if not portfolios:
        return portfolios
    
    # Create a copy
    portfolios = list(portfolios)
    
    if key == "value":
        for p in portfolios:
            if "value" not in p:
                p["_sort_value"] = _calculate_portfolio_value(p)
        # Use the calculated value for sorting
        sort_key = "_sort_value" if "_sort_value" in portfolios[0] else "value"
    else:
        sort_key = key
    
    # Insertion sort algorithm (descending order)
    for i in range(1, len(portfolios)):
        item = portfolios[i]
        j = i - 1
        
        item_value = item.get(sort_key, 0)

        while j >= 0 and portfolios[j].get(sort_key, 0) < item_value:
            portfolios[j + 1] = portfolios[j]
            j -= 1

        portfolios[j + 1] = item

    return portfolios




def manual_linear_regression(y_values):

    #Calculate slope and intercept using least squares method.

    n = len(y_values)
    
    if n == 0:
        return 0, 0
    if n == 1:
        return 0, y_values[0]
    
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(y_values) / n

    numerator = sum((xs[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean

    return slope, intercept





def recursive_portfolio_value(stocks, index=0):
    """
    Calculate total portfolio value using recursion.
    
    How It Works:
    ------------
    For stocks [A=$100, B=$200, C=$150]:
    
    Call 1: 100 + recursive(index=1)
    Call 2: 200 + recursive(index=2)
    Call 3: 150 + recursive(index=3)
    Call 4: 0 (base case)
    
    Unwinding: 0 → 150 → 350 → 450
    
    Args:
        stocks: list of stock dictionaries
        index: current position in list (default 0)
        
    Returns:
        float: Total value of all stocks
    """
    # Base case: reached end of list
    if index >= len(stocks):
        return 0.0
    
    # Calculate current stock's value
    stock = stocks[index]
    price = stock.get("purchase_price", stock.get("price", 0))
    shares = stock.get("shares", 1)
    current_value = price * shares
    
    # Recursive case: current value + sum of remaining stocks
    return current_value + recursive_portfolio_value(stocks, index + 1)
