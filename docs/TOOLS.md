# Tools

| Tool | Args | Credentials | Failure |
|---|---|---|---|
| `current_time` | `timezone` (IANA) | none | invalid zone rejected before any clock read |
| `calculator` | `expression` | none | hostile tokens / huge exponents rejected; division by zero returns `ok=False` |
| `currency_convert` | `amount`, `from_currency`, `to_currency` | none (Frankfurter) | timeout / HTTP / bad JSON → tool unavailable |
| `weather` | `city` | none (Open-Meteo) | bad city, geocode miss, HTTP errors |

Web search is not enabled unless a future official search key is configured. Tools never fetch user-supplied URLs.
