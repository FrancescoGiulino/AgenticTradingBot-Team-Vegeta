from ddgs import DDGS

results = DDGS().text("nvidia, apple, meta affected by energy crysis", max_results=10)
for r in results:
    print(r)
