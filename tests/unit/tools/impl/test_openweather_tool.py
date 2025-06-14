"""Unit tests for the OpenWeatherMapTool."""
from typing import Any, AsyncGenerator, Dict, Optional
from unittest.mock import AsyncMock

import httpx
import pytest
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.openweather import OpenWeatherMapTool


class MockWeatherKeyProvider(KeyProvider):
    plugin_id = "mock_weather_key_provider_for_owm_tests"
    async def get_key(self, key_name: str) -> Optional[str]:
        if key_name == OpenWeatherMapTool.API_KEY_NAME:
            return "fake_openweathermap_api_key"
        return None
    async def setup(self,config: Optional[Dict[str, Any]]=None): pass
    async def teardown(self): pass

@pytest.fixture()
async def openweather_tool_fixture() -> AsyncGenerator[OpenWeatherMapTool, None]:
    tool = OpenWeatherMapTool()
    await tool.setup()
    yield tool
    await tool.teardown()


@pytest.fixture()
def mock_weather_key_provider_fixture() -> MockWeatherKeyProvider:
    # Synchronous fixture returning an instance
    return MockWeatherKeyProvider()


@pytest.mark.asyncio()
async def test_openweather_tool_get_metadata(openweather_tool_fixture: AsyncGenerator[OpenWeatherMapTool, None]):
    actual_tool = await anext(openweather_tool_fixture)
    metadata = await actual_tool.get_metadata()
    assert metadata["identifier"] == "open_weather_map_tool"

@pytest.mark.asyncio()
async def test_openweather_tool_execute_success(
    openweather_tool_fixture: AsyncGenerator[OpenWeatherMapTool, None],
    mock_weather_key_provider_fixture: MockWeatherKeyProvider,
    mocker
):
    actual_tool = await anext(openweather_tool_fixture)
    actual_kp = mock_weather_key_provider_fixture
    mock_response_data = {"cod": 200, "name": "London", "sys": {"country": "GB"}, "weather": [{"main": "Clouds"}], "main": {"temp": 15.0}, "wind": {"speed": 5.5}}

    mock_get = AsyncMock(return_value=httpx.Response(200, json=mock_response_data))
    actual_tool._http_client.get = mock_get # type: ignore

    params = {"city": "London", "units": "metric"}
    result = await actual_tool.execute(params, actual_kp, context={})

    mock_get.assert_awaited_once()
    assert result["error_message"] is None
    assert result["city_name"] == "London"

@pytest.mark.asyncio()
async def test_openweather_tool_execute_api_key_missing(
    openweather_tool_fixture: AsyncGenerator[OpenWeatherMapTool, None],
    mocker
):
    actual_tool = await anext(openweather_tool_fixture)
    mock_empty_kp = mocker.AsyncMock(spec=KeyProvider)
    mock_empty_kp.get_key = AsyncMock(return_value=None)
    params = {"city": "Paris"}
    result = await actual_tool.execute(params, mock_empty_kp, context={})
    assert result["error_message"] == f"API key '{OpenWeatherMapTool.API_KEY_NAME}' is required but was not provided."

@pytest.mark.asyncio()
async def test_openweather_tool_execute_http_error(
    openweather_tool_fixture: AsyncGenerator[OpenWeatherMapTool, None],
    mock_weather_key_provider_fixture: MockWeatherKeyProvider,
    mocker
):
    actual_tool = await anext(openweather_tool_fixture)
    actual_kp = mock_weather_key_provider_fixture

    mock_request = httpx.Request("GET", OpenWeatherMapTool.API_BASE_URL)
    error_message_detail = "Invalid API key test"
    mock_response = httpx.Response(401, json={"message": error_message_detail}, request=mock_request)

    http_error = httpx.HTTPStatusError(
        message=f"Mock HTTPStatusError: {error_message_detail}", # Message for the exception itself
        request=mock_request,
        response=mock_response
    )
    actual_tool._http_client.get = AsyncMock(side_effect=http_error)

    params = {"city": "InvalidCity"}
    result = await actual_tool.execute(params, actual_kp, context={})

    # The tool's execute method constructs the error message like this:
    # f"HTTP error {e.response.status_code}: {error_body}"
    # where error_body comes from e.response.json().get("message", e.response.text)
    # In this mocked scenario, error_body will be "Invalid API key test"
    expected_error_in_result = f"HTTP error 401: {error_message_detail}"
    assert result["error_message"] == expected_error_in_result
    assert result["api_response_code"] == 401

@pytest.mark.asyncio()
async def test_openweather_tool_execute_owm_specific_error_in_200_response(
    openweather_tool_fixture: AsyncGenerator[OpenWeatherMapTool, None],
    mock_weather_key_provider_fixture: MockWeatherKeyProvider,
    mocker
):
    actual_tool = await anext(openweather_tool_fixture)
    actual_kp = mock_weather_key_provider_fixture
    mock_response_data = {"cod": "404", "message": "city not found"}
    mock_get = AsyncMock(return_value=httpx.Response(200, json=mock_response_data))
    actual_tool._http_client.get = mock_get # type: ignore

    params = {"city": "CityThatDoesNotExist"}
    result = await actual_tool.execute(params, actual_kp, context={})
    assert result["error_message"] == "OpenWeatherMap API error: city not found"
    assert result["api_response_code"] == 404
