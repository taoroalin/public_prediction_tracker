import aiohttp
import asyncio
import os
import hashlib
import pathlib
import xml.etree.ElementTree as ET
import json
import re
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def get_website_content(url: str) -> str:
    # jina.ai is a service which converst websites to markdown to make them accessible to LLMs, it helps avoid bot detection somewhat
    jina_url = f"https://r.jina.ai/{url}"
    
    cache_file = pathlib.Path(".cache/jina") / hashlib.md5(url.encode()).hexdigest()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return f.read()
    
    # If not cached, fetch from Jina AI
    async with aiohttp.ClientSession() as session:
        async with session.get(jina_url, headers={"Authorization": "Bearer jina_a43734a6afbc40328e799bb42469d974kGtj-jjWt08BLv8mQYgLaJ69ZTc0"}) as response:
            if response.status == 200:
                content = await response.text()
                # Save to cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(content)
                return content
            elif response.status == 429:
                print("Rate limit exceeded, waiting 10 seconds")
                await asyncio.sleep(10)
                return await get_website_content(url)
            elif response.status in [402, 403, 404]:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write('None')
                return None
            else:
                print(f"Failed to fetch content: HTTP {response.status}")
                return None
                raise Exception(f"Failed to fetch content: HTTP {response.status}")

async def extract_predictions(url: str) -> list:
    content = await get_website_content(url)
    if not content:
        return []
    cache_file = pathlib.Path(".cache/predictions") / hashlib.md5(url.encode()).hexdigest()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    
    
    prompt = """
Extract any predictions about the future that the authors (not quotes of other people or articles) made using their own judgement or opinions in the content pasted below.
For each prediction, provide:
1. Paraphrased prediction text, like 'There will not be a recession in H1 2019
2. The exact quote containing the prediction
3. Referenced datetime lower bound (when the prediction might start being relevant)
4. Referenced datetime upper bound (when the prediction might end being relevant)

If dates are not explicitly mentioned, infer them from context or mark as "unspecified".
Only include genuine predictions about future outcomes, not hypotheticals or opinions.

Format the output as a JSON object like this:
{
    "predictions": [
        {
            "paraphrased_prediction": "...",
            "exact_quote": "...",
            "datetime_lower_bound": "..." (YYYY-MM-DD or "unspecified"),
            "datetime_upper_bound": "..." (YYYY-MM-DD or "unspecified"),
        },
        ...
    ]
}

If no concrete predictions are found, return an empty array. Many texts won't have any predictions, which is expected, just return an empty array in that case.

Only include predictions made by post authors, not quotes from other people, subject matter experts, articles, papers, or any source other than the author of the top level post. Never include predictions made in markdown block quotes with `> `.

Only include predictions of a specific time, such as 'next year' or 'in 2025', and do not include predictions without a given time such as 'nvidia will go up' if there's no context clarifying the time frame. 

Do not include statements about official future plans, such as 'the 2030 olympics will be held in ...' or 'the tariffs will go into effect in 2025', only the author's opinion or judgement about the future.

Here are some example texts:
<example>
    s, as of now, the FDA is moving in the wrong direction and Makary has lost an ally against RFK.

In other news, the firing of FDA staff is [slowing down approvals](https://www.reuters.com/business/healthcare-pharmaceuticals/fda-staff-struggle-meet-product-review-deadlines-after-doge-layoffs-2025-03-27/), as [I predicted it would](https://marginalrevolution.com/marginalrevolution/2025/02/hire-dont-fire-at-the-fda.html).

> Mexico's economy is slowing sharply and will soon fall into recession, several economists predict, as Donald Trump's changing tariff plans cast uncertainty over the relationship with its largest trading partner.
> 
> Mexico is one of the countries most vulnerable to the US president's drive to reshore investment and close trade deficits. The country's economy was already fragile, with the government cutting spending due to a gaping budget deficit and investors spooked by its radical judicial reforms.
> 
> Mexico's GDP shrank 0.6 per cent in the fourth quarter of last year from the previous three months, while economic activity fell 0.2 per cent in January.
> 
> The central bank cut its key interest rate by 50 basis points on Thursday, warning that the economy would show weakness in the first quarter and that trade tensions posed "significant downward risks".

Here is [more from the FT](https://www.ft.com/content/e8488ad0-212e-445d-813a-eec2b6750216).
<end_example>
The above example should return an empty array, because all predictions in the text are in block quotes.
<example>
I think that the FDA's excellent arguments for conditional approval apply to human drugs as well as to (other) animal drugs and even more so when we recognize that human beings have rights and interests in making their own choices. The Promising Pathways Act would create something like conditional approval (the act calls it provisional approval) for drugs treating human diseases that are life-threatening so there is some hope that conditional approval for human drugs becomes a reality.
<end_example>
The above example should return an empty array, because it only says 'there is hope'. Expressing hope is not a prediction.
<example>
In other news, the firing of FDA staff is slowing down approvals, as I predicted it would.
<end_example>
The above example should return an empty array, because it mentions a previous prediction, and does not make a prediction about the *future* relative to the time of writing.
<example>
Monthly inflation in Argentina could fall below 10% in April, a sign that the government's policies are working, President Javier Milei said Sunday in a phone interview with LN+
<end_example>
The above example should return an empty array, because it says monthly inflation *could* fall in april, instead of saying it *will* fall in april, and it's a quote by Javier Milei, not the author.
<example>
the CMS issued new benchmarks that will allow the agency to weed out poor performers beginning in 2026.
<end_example>
The above example should return an empty array, because it's a statement of the agency's plan, not a prediction.
<example>
This prediction is from Manifold Markets. Metaculus gives similar odds to a similar question. These are serious predictions.

In a 2019 post I pointed out that expert surveys (not markets) suggested the annualized probability of a nuclear war was on the order of  ~1%–and I thought that was worryingly high. We are now at ten times that level. This is very, very bad.
<end_example>
The above example should return an empty array, because it's a prediction from Manifold Markets or Metaculus, not the author.
<example>
[posted January 29, 2021]
The contentiousness is much worse in Europe, where zero- and negative-sum thinking is the order of the day.  That is the theme of my latest Bloomberg column, here is one bit:

> In most of Europe, it’s hard to see much good news. It’s one thing not to have a vaccine. It’s far worse to turn on television or go on the internet and see people in other countries being vaccinated as their pandemics recede. Most of Europe will not be making significant vaccination progress until April, and even then shortages may remain.

> At stake is the very legitimacy of the EU. Most of the vaccination contracts were handled at the EU level, although Germany sidestepped the agreed-upon procedures and cut some deals. If the EU fails at the most significant crisis in a generation, it may not maintain much legitimacy.

And:

> When people judge how painful an experience was, they often place a high value on first and last impressions. The last impressions of the U.S. and U.K. will be pretty positive. Most of the U.S. pandemic will be over by July, even under a subpar vaccination schedule. And it may turn out that mRNA vaccines are more protective against the new strains of Covid than any alternatives….

> Many European countries may end up with fewer deaths per capita than the U.S. But at the end of the pandemic many Europeans may feel like their leaders failed them, that they suffered lockdowns for many months but received little in return. Right now vaccine politics is all about momentum, and so far only a few countries have it.

Here is a related piece by Bruno M.  And a good piece (slow to start) on what went wrong in the EU.
<end_example>
The above example should return one prediction, {
    "paraphrased_prediction": "By July 2021, most of the U.S. pandemic will be over.",
    "exact_quote": "Most of the U.S. pandemic will be over by July, even under a subpar vaccination schedule.",
    "datetime_lower_bound": "2021-07-01",
    "datetime_upper_bound": "2021-07-31",
    "url": "https://marginalrevolution.com/marginalrevolution/2021/01",
    "publish_date": "2021-01-01"
  }.
<example>

That is the topic of my latest Bloomberg column, here is one bit:

> Some of the safer locales may decide to open up, perhaps with visitor quotas. Many tourists will rush there, either occasioning a counterreaction — that is, reducing the destination’s appeal — or filling the quota very rapidly. Then everyone will resume their search for the next open spot, whether it’s Nova Scotia or Iceland. Tourists will compete for status by asking, “Did you get in before the door shut?”

> Some countries might allow visitors to only their more distant (and less desirable?) locales, enforcing movements with electronic monitoring. Central Australia, anyone? I’ve always wanted to see the northwest coast of New Zealand’s South Island.

> Some of the world’s poorer countries might pursue a “herd immunity” strategy, not intentionally, but because their public health institutions are too weak to mount an effective response to Covid-19. A year and a half from now, some of those countries likely will be open to tourism. They won’t be able to prove they are safe, but they might be fine nonetheless. They will attract the kind of risk-seeking tourist who, pre-Covid 19, might have gone to Mali or the more exotic parts of India.

And:

> laces reachable by direct flights will be increasingly attractive. A smaller aviation sector will make connecting flights more logistically difficult, and passengers will appreciate the certainty that comes from knowing they are approved to enter the country of their final destination and don’t have to worry about transfers, delays or cancellations. That will favor London, Paris, Toronto, Rome and other well-connected cities with lots to see and do. More people will want to visit a single locale and not worry about catching the train to the next city. Or they might prefer a driving tour. How about flying to Paris and then a car trip to the famous cathedrals and towns of Normandy?

Maybe. But I might start by giving Parkersburg, West Virginia, a try.
<end_example>
The above should return one prediction, because the quote which makes the prediction is clearly stated outside the quote as being from the blog post author. The quote should be "A year and a half from now, some of those countries likely will be open to tourism. They won’t be able to prove they are safe, but they might be fine nonetheless. They will attract the kind of risk-seeking tourist who, pre-Covid 19, might have gone to Mali or the more exotic parts of India."
    
The content is a concatenation of multiple blog posts. Remember, only extract predictions made by the blogger, not quoted sources.

Now, here is the actual content to extract predictions from:

"""

    publish_date = get_url_publish_date(url)
    
    try:
        # Use asyncio.to_thread to run the synchronous OpenAI API call in a separate thread
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a prediction extraction assistant. You extract concrete predictions authors made about the future from blog posts, if there are any."},
                {"role": "user", "content": prompt + content}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)["predictions"]
        for prediction in result:
            prediction["url"] = url
            prediction['publish_date'] = publish_date
        
        result = [x for x in result if x['datetime_lower_bound'] != 'unspecified' and x['datetime_upper_bound'] != 'unspecified']
        
        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        return result
    except Exception as e:
        print(f"Error extracting predictions: {e}")
        return []
    
async def validate_prediction(prediction: dict) -> bool:
    content = await get_website_content(prediction['url'])
    if not content:
        return False
    cache_file = pathlib.Path(".cache/validation") / hashlib.md5(json.dumps(prediction).encode()).hexdigest()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    prompt = f"""
Another AI tried to extract a prediction from a blog post.
Determine if this prediction meets ALL of the following requirements:
1. It's made by the blog post author (not a quote from someone else)
2. It's a genuine prediction about a future outcome (not a hypothetical or opinion)
3. It has a specific time frame mentioned, such as 'next year' or 'in 2025'
4. It's the author's opinion/judgment (not a statement about official future plans)

Prediction details:
- Paraphrased prediction: {prediction['paraphrased_prediction']}
- Exact quote: {prediction['exact_quote']}
- Time frame: From {prediction['datetime_lower_bound']} to {prediction['datetime_upper_bound']}

The content of the blog post which the prediction was extracted from is:

<content>
{content}
</content>

Remember that '> ' is used for block quotes in the content, which are not written by the blog post author.

Answer ONLY with 'Yes' or 'No'.
"""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a prediction validation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        result = 'yes' in answer.lower()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    except Exception as e:
        print(f"Error validating prediction: {e}")
        return False
    
def get_url_publish_date(url: str) -> str:
    pattern = r'https://marginalrevolution\.com/marginalrevolution/(\d{4})/(\d{2})'
    match = re.match(pattern, url)
    if not match:
        return "unspecified"
    year, month = match.groups()
    return f"{year}-{month}-01"

async def get_mr_months():
    date_archives = await get_website_content("https://marginalrevolution.com/date-archives")
    # Extract all month archive links using string matching
    
    pattern = r'https://marginalrevolution\.com/marginalrevolution/\d{4}/\d{2}'
    matches = re.findall(pattern, date_archives)
    print(matches)
    return matches


async def main():
    mr_months = await get_mr_months()
    
    predictions = []
    for month in mr_months:
        predictions.extend(await extract_predictions(month))
    print(predictions)
    validated_predictions = []
    for prediction in predictions:
        if await validate_prediction(prediction):
            validated_predictions.append(prediction)
    print(validated_predictions)
    open('validated_predictions.json', 'w').write(json.dumps(validated_predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
