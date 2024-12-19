import pprint
from googleapiclient.discovery import build
import os
import dotenv
import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from ...config import GEMINI_API_KEY, GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_CUSTOM_SEARCH_ENGINE_ID

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
dotenv.load_dotenv()
GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY')
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')

print(GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY)
print(GOOGLE_CUSTOM_SEARCH_ENGINE_ID)
def search(query):
    service = build("customsearch", "v1", developerKey=GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY)
    res = (
        service.cse()
        .list(
            q=query,
            cx=GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
        )
        .execute()
    )
    pprint.pprint(res)
    return res

search("lectures")



def main():
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build(
        "customsearch", "v1", developerKey=GOOGLE_CUSTOM_SEARCH_API_KEY
    )

    res = (
        service.cse()
        .list(
            q="lectures",
            cx="32e9bbeb5cbee467a:omuauf_lfve",
        )
        .execute()
    )
    pprint.pprint(res)


if __name__ == "__main__":
    main()