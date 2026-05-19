import urllib.request, json

tests = [
    # Very long Amazon URL with all query params
    "https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX1W1XY/ref=sr_1_1?crid=3KXYZ&keywords=iphone+15&qid=1716000000&sprefix=iphone%2Caps%2C200&sr=8-1&th=1",
    # Amazon product-reviews direct
    "https://www.amazon.in/product-reviews/B0CHX1W1XY",
    # No scheme — should auto-fix on server
    "www.amazon.in/dp/B0CHX1W1XY",
    # Flipkart long URL
    "https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm6ac6485515ae4?pid=MOBGTAGPAQNVFZZY&lid=LSTMOBGTAGPAQNVFZZYFBPVHP",
    # eBay
    "https://www.ebay.com/itm/123456789012",
    # Trustpilot
    "https://www.trustpilot.com/review/www.amazon.com",
]

for url in tests:
    body = json.dumps({"url": url}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:5000/analyse-url",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        r = urllib.request.urlopen(req, timeout=35)
        d = json.loads(r.read())
        total = d.get("total", 0)
        gp = d.get("genuine_pct", 0)
        fp = d.get("fake_pct", 0)
        print(f"OK   [{total} reviews] Genuine={gp}% Fake={fp}%  |  {url[:70]}")
    except urllib.error.HTTPError as e:
        d = json.loads(e.read())
        print(f"ERR  [{e.code}] {d.get('error','?')[:90]}  |  {url[:70]}")
    except Exception as e:
        print(f"EXC  {e}  |  {url[:70]}")
