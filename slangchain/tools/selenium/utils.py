"""Selenium Utils"""
from typing import List, Optional

import unicodedata
import re

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from slangchain.schemas import SeleniumWebElement

def unidecode(text: str) -> str:
  """ Normalize the text to NFKD form to separate characters and diacritics """

  normalized_text = unicodedata.normalize('NFKD', text)

  # Initialize an empty result string
  result = ''

  for char in normalized_text:
    # Check if the character is in the ASCII range
    if ord(char) < 128:
      result += char
    else:
      # Replace non-ASCII characters with an empty string
      result += ''

  return result


def prettify_text(text: str, limit: Optional[int] = None) -> str:
  """Prettify text by removing extra whitespace and converting to lowercase."""
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  text = unidecode(text)
  if limit:
    text = text[:limit]
  return text

def get_all_text_elements(driver: WebDriver) -> List[SeleniumWebElement]:
  """Get all text elements"""
  xpath = (
    "//*[not(self::script or self::style or"
    " self::noscript)][string-length(normalize-space(text())) > 0]"
  )

  elements = driver.find_elements(By.XPATH, xpath)
  text_elements = [
    SeleniumWebElement.from_objs(
      element_id = element.id,
      description = prettify_text(element.text)
    )
    for element in elements
      if element.text.strip() and element.is_displayed()
  ]
  return text_elements


def find_interactable_elements(driver: WebDriver) -> List[SeleniumWebElement]:
  """Find all interactable elements on the page."""
  # Extract interactable components (buttons and links)
  buttons = driver.find_elements(By.XPATH, "//button")
  links = driver.find_elements(By.XPATH, "//a")

  interactable_elements = buttons + links

  interactable_output = []
  for element in interactable_elements:
    if (
      element.is_displayed()
      and element.is_enabled()
    ):
      element_text = element.text.strip()
      if element_text and element_text not in interactable_output:
        element_text = prettify_text(element_text, 50)
        interactable_output.append(
          SeleniumWebElement.from_objs(
            element_id = element.id,
            description = element.text))
  return interactable_output

def find_parent_element_text(elem: WebElement, prettify: bool = True) -> str:
  """Find the text up to third order parent element."""
  parent_element_text = elem.text.strip()
  if parent_element_text:
    return (
      parent_element_text if not prettify else prettify_text(parent_element_text)
    )
  elements = elem.find_elements(By.XPATH, "./ancestor::*[position() <= 3]")
  for parent_element in elements:
    parent_element_text = parent_element.text.strip()
    if parent_element_text:
      return (
        parent_element_text
        if not prettify
        else prettify_text(parent_element_text)
      )
  return ""


def truncate_string_from_last_occurrence(string: str, character: str) -> str:
  """Truncate a string from the last occurrence of a character."""
  last_occurrence_index = string.rfind(character)
  if last_occurrence_index != -1:
    truncated_string = string[: last_occurrence_index + 1]
    return truncated_string
  else:
    return string
