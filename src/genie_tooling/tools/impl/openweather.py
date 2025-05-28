"""OpenWeatherMapTool: Fetches current weather information using OpenWeatherMap API."""
import logging
from typing import Any, Dict, Optional, Union

import httpx  # Requires: poetry add httpx

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class OpenWeatherMapTool(Tool):
    identifier: str = "open_weather_map_tool"
    plugin_id: str = "open_weather_map_tool" # Matches identifier
    API_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    API_KEY_NAME = "OPENWEATHERMAP_API_KEY" # Standardized key name

    _http_client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes an async HTTP client for reuse."""
        self._http_client = httpx.AsyncClient(timeout=10.0) # Default 10s timeout
        logger.debug("OpenWeatherMapTool: HTTP client initialized.")

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Current Weather (OpenWeatherMap)",
            "description_human": "Fetches the current weather conditions for a specified city using the OpenWeatherMap API. Requires an API key.",
            "description_llm": "WeatherInfo: Get current weather for a city. Args: city (str, required, e.g., 'London, UK'), units (str, optional, 'metric' for Celsius or 'imperial' for Fahrenheit, default: 'metric').",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city and optionally country code (e.g., 'Paris, FR', 'Tokyo')."
                    },
                    "units": {
                        "type": "string",
                        "description": "Units for temperature. 'metric' for Celsius, 'imperial' for Fahrenheit.",
                        "enum": ["metric", "imperial"],
                        "default": "metric"
                    }
                },
                "required": ["city"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "city_name": {"type": ["string", "null"], "description": "Resolved name of the city from the API response."},
                    "country": {"type": ["string", "null"], "description": "Country code (e.g., GB, US)."},
                    "temperature_celsius": {"type": ["number", "null"], "description": "Current temperature in Celsius (if units=metric)."},
                    "temperature_fahrenheit": {"type": ["number", "null"], "description": "Current temperature in Fahrenheit (if units=imperial)."},
                    "feels_like_celsius": {"type": ["number", "null"], "description": "'Feels like' temperature in Celsius (if units=metric)."},
                    "feels_like_fahrenheit": {"type": ["number", "null"], "description": "'Feels like' temperature in Fahrenheit (if units=imperial)."},
                    "condition": {"type": ["string", "null"], "description": "Brief weather condition (e.g., 'Clear sky', 'Rain')."},
                    "description": {"type": ["string", "null"], "description": "More detailed weather condition description."},
                    "humidity_percent": {"type": ["integer", "null"], "description": "Humidity percentage (0-100)."},
                    "wind_speed_mps": {"type": ["number", "null"], "description": "Wind speed in meters per second (if units=metric)."},
                    "wind_speed_mph": {"type": ["number", "null"], "description": "Wind speed in miles per hour (if units=imperial)."},
                    "pressure_hpa": {"type": ["integer", "null"], "description": "Atmospheric pressure in hPa."},
                    "visibility_meters": {"type": ["integer", "null"], "description": "Visibility in meters."},
                    "sunrise_timestamp": {"type": ["integer", "null"], "description": "Sunrise time, UNIX UTC timestamp."},
                    "sunset_timestamp": {"type": ["integer", "null"], "description": "Sunset time, UNIX UTC timestamp."},
                    "timezone_offset_seconds": {"type": ["integer", "null"], "description": "Shift in seconds from UTC."},
                    "api_response_code": {"type": ["integer", "null"], "description": "HTTP status code from OpenWeatherMap API if an error specific to API call occurs."},
                    "error_message": {"type": ["string", "null"], "description": "Error message if fetching weather data failed."}
                },
                # No "required" on output fields, as they might be null on error or based on units
            },
            "key_requirements": [{"name": self.API_KEY_NAME, "description": "Your personal API key for OpenWeatherMap."}],
            "tags": ["weather", "location", "api", "external-data"],
            "version": "1.2.0",
            "cacheable": True,
            "cache_ttl_seconds": 10 * 60 # Cache weather data for 10 minutes
        }

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[float, str, int, None]]:
        if not self._http_client:
            # This should ideally not happen if setup is called correctly.
            logger.error("OpenWeatherMapTool: HTTP client not initialized. Please ensure setup() was called.")
            return {"error_message": "Tool not properly initialized: HTTP client missing."}

        api_key = await key_provider.get_key(self.API_KEY_NAME)
        if not api_key:
            logger.warning(f"OpenWeatherMapTool: API key '{self.API_KEY_NAME}' not found via KeyProvider.")
            return {"error_message": f"API key '{self.API_KEY_NAME}' is required but was not provided."}

        city = params["city"]
        units = params.get("units", "metric") # Default to metric as per schema
        query_params = {"q": city, "appid": api_key, "units": units}

        try:
            logger.debug(f"OpenWeatherMapTool: Fetching weather for city='{city}', units='{units}'")
            response = await self._http_client.get(self.API_BASE_URL, params=query_params)

            # Check for OpenWeatherMap specific error codes within a 200 response
            if response.status_code == 200:
                data = response.json()
                if str(data.get("cod")) != "200": # OWM uses string "200" for success in `cod`
                    owm_error_message = data.get("message", "Unknown OpenWeatherMap API error.")
                    logger.warning(f"OpenWeatherMap API error for '{city}': {owm_error_message} (cod: {data.get('cod')})")
                    return {
                        "error_message": f"OpenWeatherMap API error: {owm_error_message}",
                        "api_response_code": int(data.get("cod", 500)) if str(data.get("cod","500")).isdigit() else 500
                    }

                # Successful response, parse data
                main_data = data.get("main", {})
                weather_info = data.get("weather", [{}])[0] # Get first weather condition
                wind_data = data.get("wind", {})
                sys_data = data.get("sys", {})

                output = {
                    "city_name": data.get("name"),
                    "country": sys_data.get("country"),
                    "condition": weather_info.get("main"),
                    "description": weather_info.get("description"),
                    "humidity_percent": main_data.get("humidity"),
                    "pressure_hpa": main_data.get("pressure"),
                    "visibility_meters": data.get("visibility"),
                    "sunrise_timestamp": sys_data.get("sunrise"),
                    "sunset_timestamp": sys_data.get("sunset"),
                    "timezone_offset_seconds": data.get("timezone"),
                    "error_message": None,
                    "api_response_code": 200
                }

                if units == "metric":
                    output["temperature_celsius"] = main_data.get("temp")
                    output["feels_like_celsius"] = main_data.get("feels_like")
                    output["wind_speed_mps"] = wind_data.get("speed")
                elif units == "imperial":
                    output["temperature_fahrenheit"] = main_data.get("temp")
                    output["feels_like_fahrenheit"] = main_data.get("feels_like")
                    output["wind_speed_mph"] = wind_data.get("speed")

                logger.info(f"OpenWeatherMapTool: Successfully fetched weather for '{city}'. Temp: {main_data.get('temp')}{'C' if units=='metric' else 'F'}")
                return output

            else: # HTTP error (4xx, 5xx)
                response.raise_for_status() # This will raise an HTTPStatusError
                # Should not be reached if raise_for_status works, but as a fallback:
                return {"error_message": f"HTTP error {response.status_code}", "api_response_code": response.status_code}


        except httpx.HTTPStatusError as e:
            error_body = ""
            try: # Try to get more details from response body
                error_body = e.response.json().get("message", e.response.text)
            except Exception:
                error_body = e.response.text
            logger.error(f"OpenWeatherMapTool: HTTP status error for '{city}'. Status: {e.response.status_code}. Response: {error_body}", exc_info=True)
            return {
                "error_message": f"HTTP error {e.response.status_code}: {error_body}",
                "api_response_code": e.response.status_code
            }
        except httpx.RequestError as e: # Covers network errors, timeouts, etc.
            logger.error(f"OpenWeatherMapTool: Request error for '{city}'. Error: {e}", exc_info=True)
            return {"error_message": f"Network or request error: {str(e)}"}
        except Exception as e: # Catch-all for unexpected issues like JSON parsing
            logger.error(f"OpenWeatherMapTool: Unexpected error for '{city}'. Error: {e}", exc_info=True)
            return {"error_message": f"An unexpected error occurred: {str(e)}"}

    async def teardown(self) -> None:
        """Closes the async HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.debug("OpenWeatherMapTool: HTTP client closed.")
