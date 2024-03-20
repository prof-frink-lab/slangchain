"""Selenium tool class"""
from typing import List, Optional, Dict, Literal, Any

import sys
import traceback
import time
import logging
import re
import base64
from math import sqrt
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
import random
import validators

from pydantic import BaseModel, Extra, Field, validator

from selenium.common.exceptions import WebDriverException

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.remote.webdriver import WebDriver

from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains

from slangchain.schemas import (
  SeleniumWebElement, SeleniumLinkWebElement, Base64ImageStringExplainedImage
)
from slangchain.chains.image_explainer.base import Base64ImageStringExplainerChain
from slangchain.tools.selenium.utils import (
  prettify_text,
  find_parent_element_text
)

logger = logging.getLogger()

RESULT_KEY =  "result"
RESULT_STATUS_KEY =  "result status"

WEB_ELEMENTS_KEY = "web_elements"
URL_KEY = "url"
CONTENT_KEY = "content"
BUTTON_KEY = "button"
BUTTONS_KEY = "buttons"
LINK_KEY = "link"
LINKS_KEY = "links"
FORM_INPUT_KEY = "form_input"
FORM_INPUTS_KEY = "form_inputs"


class SeleniumWrapper(BaseModel):
  """Selenium Wrapper class"""

  browser_headless_flag: bool
  driver_timeout: int
  driver: WebDriver

  text_web_elements: Optional[List[WebElement]] = Field(default = [])
  text_elements: Optional[List[SeleniumWebElement]] = Field(default = [])

  link_web_elements: Optional[List[WebElement]] = Field(default = [])
  link_elements: Optional[List[SeleniumWebElement]] = Field(default = [])

  button_web_elements: Optional[List[WebElement]] = Field(default = [])
  button_elements: Optional[List[SeleniumWebElement]] = Field(default = [])

  form_web_elements: Optional[List[WebElement]] = Field(default = [])
  form_elements: Optional[List[SeleniumWebElement]] = Field(default = [])

  interactable_output: Optional[str] = Field(default = "")

  max_description_length: Optional[int] = Field(default=200)

  model_describer_flag: Optional[bool] = Field(default = False)
  model_describer_name: Optional[str] = Field(default=None)
  temperature: Optional[float] = Field(default = 0.0)
  model_describer_max_tokens: Optional[int] = Field(default=1024)

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

  @validator("model_describer_name")
  def validate_model_name(cls, v: str) -> str:
    """Validate that model_name is allowed."""
    if not v.startswith("claude-3") and not v.startswith("gpt-4-vision"):
      raise ValueError(f"model_describer_name {v} not valid.")
    return v


  def describe_website(self, url: Optional[str] = None) -> str:
    """Describe the website."""
    current_url = url if url else self.driver.current_url
    if url:
      try:
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(url)
      except Exception as exp:
        logger.error("convert error: %s", exp)

    # Let driver wait for website to load
    time.sleep(1)  # Wait for website to load

    try:
      # Extract main content
      main_content = self._get_website_main_content()
    except WebDriverException as e:
      result_status = (
        f"Error describing website: {current_url}, message: {e.msg}"
      )
      return result_status

    return f"url: {current_url}\n{main_content}\n" if main_content else f"url: {current_url}\n"

  def previous_webpage(self) -> str:
    """Go back in browser history."""
    self.driver.back()
    return self.describe_website()

  def _find_text_web_elements(self) -> List[WebElement]:
    """fine text web elements"""
    self.text_web_elements = self.driver.find_elements(
      By.XPATH,
      ( "//*[not(self::script or self::style or"
        " self::noscript)][string-length(normalize-space(text())) > 0]"),
    )
    return self.text_web_elements

  def _find_text(self) -> List[SeleniumWebElement]:
    """find text"""
    self.text_web_elements = self._find_text_web_elements()

    self.text_elements = [
      SeleniumWebElement.from_objs(
        element_id = element.id,
        description = prettify_text(element.text, self.max_description_length),
        element_type = CONTENT_KEY
      )
      for element in self.text_web_elements
        if element.text.strip()
    ]
    return self.text_elements

  def _get_ai_website_description(self) -> List[SeleniumWebElement]:
    """_get_ai_website_description"""

    base64_image = self.driver.get_screenshot_as_base64()

    explainer_chain = Base64ImageStringExplainerChain(
      model_name = self.model_describer_name,
      temperature = self.temperature,
      max_tokens = self.model_describer_max_tokens,
      prompt = 
        ("Your description must be detailed enough"
         " so that you are able to discern between, text, link, form amd button elements")
    )

    result_dict: Dict[str, Any] = explainer_chain.invoke(
      { "base64_image_string": base64_image, "source": self.driver.current_url })

    explainer_result: Base64ImageStringExplainedImage = result_dict["result"]

    self.text_elements = [
      SeleniumWebElement.from_objs(
        element_id = "description",
        description = explainer_result.explanation,
        element_type = CONTENT_KEY
      )
    ]

    return self.text_elements

  def _get_website_main_content(self) -> str:
    """ get website main content """
    if self.model_describer_flag:
      self.text_elements = self._get_ai_website_description()
    else:
      self.text_elements = self._find_text()

    description = ' '.join([i.description for i in self.text_elements])

    return description

  def _find_link_web_elements(self) -> List[WebElement]:
    """find link web elements"""
    self.link_web_elements = self.driver.find_elements(
      By.XPATH,
      "//a | //input[@type='checkbox']",
    )
    return self.link_web_elements

  def _find_links(self) -> List[SeleniumLinkWebElement]:
    """ Extract interactable components (links) """
    self.link_web_elements = self._find_link_web_elements()

    self.link_elements: List[SeleniumWebElement] = []

    for element in self.link_web_elements:
      if not validators.url(element.get_attribute("href") if element.get_attribute("href") else ""):
        continue
      link_text = find_parent_element_text(element)
      link_text = prettify_text(link_text, self.max_description_length)

      if (link_text and element.is_displayed() and element.is_enabled()):
        self.link_elements.append(
          SeleniumLinkWebElement.from_objs(
            element_id = element.id,
            description = link_text,
            element_type = LINK_KEY,
            url = element.get_attribute("href"),
          ))

    return self.link_elements


  def _find_link_elements(self, url: Optional[str] = None) -> List[SeleniumLinkWebElement]:
    """ Extract interactable components (links) """

    self.link_elements: List[SeleniumWebElement] = []
    if url and url != self.driver.current_url and url.startswith("http"):
      try:
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(url)
        # Let driver wait for website to load
        time.sleep(1)  # Wait for website to load
      except WebDriverException:
        return self.link_elements

    self.link_web_elements = self._find_link_web_elements()

    for element in self.link_web_elements:
      if not validators.url(element.get_attribute("href") if element.get_attribute("href") else ""):
        continue
      link_text = find_parent_element_text(element)
      link_text = prettify_text(link_text, self.max_description_length)

      if (link_text and element.is_displayed() and element.is_enabled()):
        self.link_elements.append(
          SeleniumLinkWebElement.from_objs(
            element_id = element.id,
            description = link_text,
            element_type = LINK_KEY,
            url = element.get_attribute("href"),
          ))

    return self.link_elements


  def find_links(self, url: Optional[str] = None) -> str:
    """Find links on the website."""

    self.link_elements = self._find_link_elements(url)

    return '\n'.join(
      [(f"element_id: {i.element_id},"
        f" description: {i.description},"
        f" type: {i.element_type},"
        f" url: {i.url}") for i in self.link_elements])


  def _find_web_element_buttons(self) -> List[WebElement]:
    self.button_web_elements = self.driver.find_elements(
      By.XPATH,
      "//button | //div[@role='button']",
    )
    return self.button_web_elements


  def _find_button_elements(self, url: Optional[str] = None) -> List[SeleniumWebElement]:
    """ Extract interactable components (buttons) """

    self.button_elements: List[SeleniumWebElement] = []
    if url and url != self.driver.current_url and url.startswith("http"):
      try:
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(url)
        # Let driver wait for website to load
        time.sleep(1)  # Wait for website to load
      except WebDriverException:
        return self.button_elements

    self.button_web_elements = self._find_web_element_buttons()

    for element in self.button_web_elements:
      if validators.url(element.get_attribute("href") if element.get_attribute("href") else ""):
        continue
      button_text = find_parent_element_text(element)
      button_text = prettify_text(button_text, self.max_description_length)

      if (button_text and element.is_displayed() and element.is_enabled()):
        self.button_elements.append(
          SeleniumWebElement.from_objs(
            element_id = element.id,
            description = button_text,
            element_type = BUTTON_KEY
          ))

    return self.button_elements

  def find_buttons(self, url: Optional[str] = None) -> str:
    """Find buttons on the website."""

    self.button_elements = self._find_button_elements(url)

    return '\n'.join(
      [(f"element_id: {i.element_id},"
        f" description: {i.description},"
        f" type: {i.element_type}") for i in self.button_elements])


  def click_button_by_element_id(self, element_id: str) -> str:
    """ check if the button text is url """

    result_status = ""

    if validators.url(element_id):
      return self.describe_website(element_id)

    self.driver.switch_to.window(self.driver.window_handles[-1])
    # If there are string surrounded by double quotes, extract them
    if element_id.count('"') > 1:
      try:
        element_id = re.findall(r'"([^"]*)"', element_id)[0]
      except IndexError:
        # No text surrounded by double quotes
        pass

    try:
      self.button_web_elements: List[WebElement] = self._find_web_element_buttons()

      if not self.button_web_elements:
        result_status = \
          "No interactable buttons were found on the website."

      if self.button_web_elements:

        selected_element = None

        for element in self.button_web_elements:
          if (element.is_displayed() and element.is_enabled() and (
            element_id == element.id)):

            selected_element = element
            break

        if not selected_element:
          result_status = (
            f"No interactable button found with element_id: {element_id}. Double"
            " check the button element_id and try again. Available buttons:\n"
            f"{self.find_buttons()}")

        if selected_element:
          if not self.browser_headless_flag:
            self.driver.execute_script(
              "arguments[0].scrollIntoView();", selected_element
            )
            time.sleep(0.5)  # Allow some time for the page to settle
            # Scroll the element into view and Click the element using JavaScript
          before_content = self._find_text()
          actions = ActionChains(self.driver)
          actions.move_to_element(selected_element).click().perform()

          self.driver.switch_to.window(self.driver.window_handles[-1])

          after_content = self._find_text()
          if before_content == after_content:
            result_status = (
              f"Clicked on interactable element {selected_element.id}"
              " but nothing changed on the website."
              " Available buttons:\n"
              f"{self.find_buttons()}"
            )
          else:
            result_status = (
              f"Clicked on interactable element {selected_element.id}"
              " and website was changed."
            )

    except WebDriverException as e:
      result_status = (
        f"Error clicking button with element_id '{element_id}', message: {e.msg}"
        " Double check the button element_id and try again. Available buttons:\n"
        f"{self.find_buttons()}"
      )

    return (
      f"{result_status}\n"
      f"{self.describe_website()}\n"
    )


  def click_button_by_description(self, description: str) -> str:
    """ check if the button text is url """

    result_status = ""

    if validators.url(description):
      return self.describe_website(description)

    self.driver.switch_to.window(self.driver.window_handles[-1])
    # If there are string surrounded by double quotes, extract them
    if description.count('"') > 1:
      try:
        description = re.findall(r'"([^"]*)"', description)[0]
      except IndexError:
        # No text surrounded by double quotes
        pass

    try:
      self.button_web_elements: List[WebElement] = self._find_web_element_buttons()

      if not self.button_web_elements:
        result_status = \
          "No interactable buttons found on the website."

      if self.button_web_elements:

        selected_element = None

        for element in self.button_web_elements:
          text = find_parent_element_text(element)
          button_text = prettify_text(text, self.max_description_length)
          if (element.is_displayed() and element.is_enabled() and (
            description == button_text
              or
            (button_text in description))):

            selected_element = element
            break

        if not selected_element:
          result_status = (
            f"No interactable element found with description: {button_text}. Double"
            f" check the button description and try again. Available buttons:\n"
            f"{self.find_buttons()}")

        if selected_element:
          if not self.browser_headless_flag:
            self.driver.execute_script(
              "arguments[0].scrollIntoView();", selected_element
            )
            time.sleep(0.5)  # Allow some time for the page to settle
            # Scroll the element into view and Click the element using JavaScript
          before_content = self._find_text()
          actions = ActionChains(self.driver)
          actions.move_to_element(selected_element).click().perform()

          self.driver.switch_to.window(self.driver.window_handles[-1])

          after_content = self._find_text()

          if before_content == after_content:
            result_status = (
              f"Clicked interactable button {button_text}"
              " but nothing changed on the website. Available buttons:\n"
              f"{self.find_buttons()}"
            )
          else:
            result_status =  (
              f"Clicked interactable button {button_text}"
              f" and changed on the website was detected."
            )
    except WebDriverException as e:
      result_status = (
          f"Error clicking button with text '{description}', message: {e.msg}"
          " Double check the button element_id and try again. Available buttons:\n"
          f"{self.find_buttons()}")
    return (
      f"{result_status}\n"
      f"{self.describe_website()}\n"
    )


  def click_link_by_element_id(self, element_id: str) -> str:
    """ check if the button text is url """

    result_status = ""

    if validators.url(element_id):
      return self.describe_website(element_id)

    self.driver.switch_to.window(self.driver.window_handles[-1])
    # If there are string surrounded by double quotes, extract them
    if element_id.count('"') > 1:
      try:
        element_id = re.findall(r'"([^"]*)"', element_id)[0]
      except IndexError:
        # No text surrounded by double quotes
        pass

    try:
      self.link_web_elements: List[WebElement] = self._find_link_web_elements()

      if not self.link_web_elements:
        result_status = \
          "No interactable links found in the website. Try another website."

      if self.link_web_elements:

        selected_element = None

        for element in self.link_web_elements:
          if (element.is_displayed() and element.is_enabled() and (
            element_id == element.id)):

            selected_element = element
            break

        if not selected_element:
          result_status = (
              f"No interactable element found with element_id: {element_id}. Double"
              f" check the link text and try again. Available links:\n"
              f"{self.find_links()}")

        if selected_element:
          if not self.browser_headless_flag:
            self.driver.execute_script(
              "arguments[0].scrollIntoView();", selected_element
            )
            time.sleep(0.5)  # Allow some time for the page to settle
            # Scroll the element into view and Click the element using JavaScript
          before_content = self._find_text()
          actions = ActionChains(self.driver)
          actions.move_to_element(selected_element).click().perform()

          self.driver.switch_to.window(self.driver.window_handles[-1])

          after_content = self._find_text()
          if before_content == after_content:
            result_status = (
              "Clicked interactable element but nothing changed on the website."
              " Available links:\n"
              f"{self.find_links()}"
            )
          else:
            result_status = (
              f"Clicked interactable element {selected_element.id}"
              f" and changed on the website was detected."
            )
    except WebDriverException as e:
      result_status = (
          f"Error clicking link with element_id '{element_id}', message: {e.msg}"
          " Double check the links element_id and try again. Available links:\n"
          f"{self.find_links()}")

    return (
      f"{result_status}\n"
      f"{self.describe_website()}\n"
    )


  def click_link_by_description(self, description: str) -> str:
    """ check if the button text is url """

    result_status = ""

    if validators.url(description):
      return self.describe_website(description)

    self.driver.switch_to.window(self.driver.window_handles[-1])
    # If there are string surrounded by double quotes, extract them
    if description.count('"') > 1:
      try:
        description = re.findall(r'"([^"]*)"', description)[0]
      except IndexError:
        # No text surrounded by double quotes
        pass

    try:
      self.link_web_elements: List[WebElement] = self._find_link_web_elements()

      if not self.link_web_elements:
        result_status = \
          "No interactable links found in the website."

      if self.link_web_elements:

        selected_element = None

        for element in self.link_web_elements:
          text = find_parent_element_text(element)
          link_text = prettify_text(text, self.max_description_length)
          if (element.is_displayed() and element.is_enabled() and (
            description == link_text
              or
            (link_text in description))):

            selected_element = element
            break

        if not selected_element:
          result_status = (
            f"No interactable link found with description: {link_text}. Double"
            f" check the link description and try again. Available links:"
            f"{self.find_links()}")

        if selected_element:
          if not self.browser_headless_flag:
            self.driver.execute_script(
              "arguments[0].scrollIntoView();", selected_element
            )
            time.sleep(0.5)  # Allow some time for the page to settle
            # Scroll the element into view and Click the element using JavaScript
          before_content = self._find_text()
          actions = ActionChains(self.driver)
          actions.move_to_element(selected_element).click().perform()

          self.driver.switch_to.window(self.driver.window_handles[-1])

          after_content = self._find_text()
          if before_content == after_content:
            result_status = (
              f"Clicked interactable link {link_text} but nothing changed on the website."
              " Available links:\n"
              f"{self.find_links()}"
            )
          else:
            result_status = (
              f"Clicked interactable link {link_text} and the website changed."
            )
    except WebDriverException as e:
      result_status = (
        f"Error clicking link with text '{description}', message: {e.msg}"
          " Double check the links description and try again. Available links:\n"
          f"{self.find_links()}")
    return (
      f"{result_status}\n"
      f"{self.describe_website()}\n"
    )


  def _find_form_web_elements(self) -> List[WebElement]:
    """find form web elements"""
    self.form_web_elements = self.driver.find_elements(By.XPATH, "//textarea | //input")
    self.form_web_elements = [
      i for i in self.form_web_elements if (i.is_displayed() and i.is_enabled())]
    return self.form_web_elements


  def _find_form_elements(self, url: Optional[str] = None) -> List[SeleniumWebElement]:
    """Find form fields on the website."""
    self.form_elements: List[SeleniumWebElement] = []
    if url and url != self.driver.current_url and url.startswith("http"):
      try:
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(url)
        # Let driver wait for website to load
        time.sleep(1)  # Wait for website to load
      except WebDriverException:
        return self.form_elements

    self.form_web_elements = self._find_form_web_elements()
    for element in self.form_web_elements:
      label_txt = ""

      if element.get_attribute("name"):
        label_txt += element.get_attribute("name") + " "
      if element.get_attribute("aria-label"):
        label_txt += element.get_attribute("aria-label") + " "
      if find_parent_element_text(element):
        label_txt += find_parent_element_text(element)

      if (
        label_txt
        and "\n" not in label_txt
        and label_txt not in [i.dict() for i in self.form_elements]
      ):
        label_txt = prettify_text(label_txt, self.max_description_length)
        self.form_elements.append(SeleniumWebElement.from_objs(
          element_id = element.id,
          description = label_txt,
          element_type = FORM_INPUT_KEY
        ))
    return self.form_elements


  def find_forms(self, url: Optional[str] = None) -> str:
    """Find form inputs on the website."""

    result_status = ""

    self.form_elements = self._find_form_elements(url)
    result_status = (
      "Form elements found." 
    )

    return f'{result_status}\n' + '\n'.join(
      [(f"element_id: {i.element_id},"
        f" description: {i.description},"
        f" type: {i.element_type}") for i in self.form_elements])


  def fill_out_form(self, form_input: Optional[str] = None, **kwargs: Any) -> str:
    """fill out form by form field name and input name"""

    result_status = ""
    form_input_dict : Dict[str, Any] = {}


    for form_input_elem in form_input.split("|"):
      form_input_elems = form_input_elem.split("=")
      if len(form_input_elems) == 2:
        form_input_dict[form_input_elems[0]] = form_input_elems[-1]


    filled_element = None

    try:
      for element in self._find_form_web_elements():
        for key, value in form_input_dict.items():  # type: ignore
          if key == element.id:
            if not self.browser_headless_flag:
              # Scroll the element into view
              self.driver.execute_script(
                "arguments[0].scrollIntoView();", element
              )
              time.sleep(0.5)  # Allow some time for the page to settle
            try:
              # Try clearing the input field
              element.send_keys(Keys.CONTROL + "a")
              element.send_keys(Keys.DELETE)
              element.clear()
            except WebDriverException:
              pass
            element.send_keys(value)  # type: ignore
            filled_element = element
            break

      if not filled_element:
        result_status = (
          f"Cannot find form with input: {form_input_dict.keys()}."  # type: ignore
          f" Available form inputs:\n {self.find_forms()}"
        )
        return (
          f"{result_status}\n"
          f"url: {self.driver.current_url}\n"
        )

      before_content = self._find_text()
      filled_element.send_keys(Keys.RETURN)
      after_content = self._find_text()
      if before_content != after_content:
        result_status = (
          f"Successfully filled out form with input: {form_input_dict}, "
          f" website changed after filling out form."
        )

      else:
        result_status =  (
          f"Successfully filled out form with input: {form_input_dict}, but"
          " website did not change after filling out form.")
    except WebDriverException as e:
      result_status = (
        f"Error filling out form with input {form_input_dict}, message: {e.msg}"
        " Double check the form element_id and try again. Available form:\n"
        f"{self.find_forms()}")

    return (
      f"{result_status}\n"
      f"{self.describe_website()}\n"
    )


  def scroll(self, direction: str) -> str:
    """ Get the height of the current window """

    result_status = ""

    window_height = self.driver.execute_script("return window.innerHeight")
    if direction == "up":
      window_height = -window_height

    # Scroll by 1 window height
    self.driver.execute_script(f"window.scrollBy(0, {window_height})")

    result_status = f"Scroll {direction} successful"

    return (
      f"{result_status}\n"
      f"{self.describe_website()}\n"
    )


  def solve_recaptcha(self, url: Optional[str] = None) -> str:
    """Solve recaptcha"""
    current_url = url if url else self.driver.current_url
    if url:
      try:
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(current_url)
      except Exception as exp:
        logger.error("convert error: %s", exp)

    solver = GoogleRecaptchaWrapper(
      driver = self.driver,
      driver_timeout = self.driver_timeout,
      model_describer_name = self.model_describer_name
    )
    result = solver.process_recaptcha()
    return result


  def find_interactable_components(self, url: Optional[str] = None) -> str:
    """find interactable components."""
    main_content = ""

    current_url = url if url else self.driver.current_url
    if url:
      try:
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(url)
      except Exception as exp:
        logger.error("convert error: %s", exp)

    # Let driver wait for website to load
    time.sleep(1)  # Wait for website to load

    try:
      main_content += f"\nAvailable Links:\n {self.find_links()}\n"
      main_content += f"\nAvailable Buttons:\n {self.find_buttons()}\n"
      main_content += f"\nAvailable Forms:\n {self.find_forms()}\n"
    except WebDriverException as e:
      result_status = (
        f"Error describing website: {current_url}, message: {e.msg}"
      )
      return result_status

    return f"url: {current_url}\n{main_content}\n" if main_content else f"url: {current_url}\n"


  @classmethod
  def from_parameters(
    cls,
    browser: Literal["chrome", "firefox"] = "chrome",
    browser_executable_path: Optional[str] =  None,
    chrome_binary_path: Optional[str] =  None,
    browser_headless_flag: Optional[bool] =  True,
    browser_load_images_flag: Optional[bool] = True,
    browser_disable_javascript_flag: Optional[bool] = False,
    max_description_length: Optional[int] = 200,
    driver_timeout: Optional[int] = 20,
    window_width: Optional[int] = 1920,
    window_height: Optional[int] = 1080,
    model_describer_flag: Optional[bool] = False,
    model_describer_name: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    model_describer_max_tokens: Optional[int] = 1024):
    """from parameters"""

    driver: WebDriver = None

    if browser.lower() == "chrome":

      chrome_options = ChromeOptions()

      chrome_options.add_argument((
        "--user-agent=Mozilla/5.0 (Linux; Android 6.0;"
        " HTC One M9 Build/MRA58K) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/52.0.2743.98 Mobile Safari/537.36"))
      chrome_options.add_argument('--no-sandbox')
      chrome_options.add_argument('--single-process')
      chrome_options.add_argument('--disable-dev-shm-usage')
      chrome_options.add_argument("--disable-blink-features=AutomationControlled")
      # Exclude the collection of enable-automation switches
      chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
      # Turn-off userAutomationExtension
      chrome_options.add_experimental_option("useAutomationExtension", False)

      browser_load_images_flag_str = f"{browser_load_images_flag}".lower()
      chrome_options.add_argument(f'--blink-settings=imagesEnabled={browser_load_images_flag_str}')

      if browser_disable_javascript_flag:
        chrome_options.add_experimental_option(
          "prefs",{'profile.managed_default_content_settings.javascript': 2})

      if chrome_binary_path:
        chrome_options.binary_location = chrome_binary_path

      if browser_headless_flag:
        chrome_options.add_argument("--headless")

      if browser_executable_path is None:
        driver =  Chrome(options=chrome_options)
      else:
        service = ChromeService(executable_path = browser_executable_path)
        driver = Chrome(
          service = service, options = chrome_options)

    elif browser.lower() == "firefox":

      firefox_options = FirefoxOptions()
      if browser_headless_flag:
        firefox_options.add_argument("--headless")
      if browser_executable_path is None:

        driver = Firefox(options=firefox_options)

      else:

        driver = Firefox(
          service = FirefoxService(executable_path = browser_executable_path),
          options = firefox_options
        )

    else:

      raise ValueError("Invalid browser specified. Use 'chrome' or 'firefox'.")

    driver.execute_script(
      "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")


    driver.set_window_size(width = window_width, height = window_height)
    driver.set_page_load_timeout(driver_timeout)

    return cls(
      browser_headless_flag = browser_headless_flag,
      driver_timeout = driver_timeout,
      driver = driver,
      max_description_length = max_description_length,
      model_describer_flag = model_describer_flag,
      model_describer_name = model_describer_name,
      temperature = temperature,
      model_describer_max_tokens = model_describer_max_tokens
    )


class GoogleRecaptchaWrapper(BaseModel):
  """Google Captcha Wrapper class"""

  driver: WebDriver
  driver_timeout: Optional[int] = Field(default = 10)

  model_describer_name: Optional[str] = Field(default = None)
  temperature: Optional[float] = Field(default = 0.0)
  model_describer_max_tokens: Optional[int] = Field(default = 1024)
  tile_by_tile_inference_flag: Optional[bool] = Field(default = False)

  not_robot_iframe_element: Optional[WebElement] = Field(default = None)
  recaptcha_checkbox_border_element: Optional[WebElement] = Field(default = None)
  recaptcha_iframe_element: Optional[WebElement] = Field(default = None)
  iframe_location: Optional[Dict[str, Any]] = Field(default = None)
  iframe_size: Optional[Dict[str, Any]] = Field(default = None)
  image_tile_elements: Optional[List[WebElement]] = Field(default = [])
  submit_button_element: Optional[WebElement] = Field(default = None)

  instructions: Optional[str] = Field(default = '')
  status_message: Optional[str] = Field(default = '')
  selected_image_tile_idxs: Optional[List[int]] = Field(default = '')
  recaptcha_label: Optional[str] = Field(default = '')
  image_element: Optional[WebElement] = Field(default = None)
  base64_image: Optional[str] = Field(default = None)
  image_tile_selections: Optional[List[int]] = Field(default = [])

  text_elements: Optional[List[SeleniumWebElement]] = Field(default = [])

  low_click_delay: float = 0.06
  high_click_delay: float = 0.1

  low_loading_image_delay: float = 2.0
  high_loading_image_delay: float = 4.0

  low_loading_images_delay: float = 4.5
  high_loading_images_delay: float = 7.0

  brightness: Optional[float] = Field(default=1.0)
  contrast: Optional[float] = Field(default=1.0)
  sharpness: Optional[float] = Field(default=1.0)

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

  @validator("model_describer_name")
  def validate_model_name(cls, v: str) -> str:
    """Validate that model_name is allowed."""
    if not v.startswith("claude-3") and not v.startswith("gpt-4-vision"):
      raise ValueError(f"model_describer_name {v} not valid.")
    return v

  def _find_not_robot_iframe(self) -> WebElement:
    """find not robot iframe"""

    captcha_xpath = \
      "//iframe[starts-with(@name, 'a-') and starts-with(@src, 'https://www.google.com/recaptcha')]"

    self.not_robot_iframe_element = WebDriverWait(self.driver, self.driver_timeout).until(
      expected_conditions.presence_of_element_located((By.XPATH, captcha_xpath))
    )

    return self.not_robot_iframe_element

  def _find_recaptcha_checkbox_border_element(self):
    self.not_robot_iframe_element = self._find_not_robot_iframe()
    self.driver.switch_to.frame(self.not_robot_iframe_element)
    self.recaptcha_checkbox_border_element = WebDriverWait(self.driver, self.driver_timeout).until(
      expected_conditions.element_to_be_clickable(
        (By.CSS_SELECTOR, "div.recaptcha-checkbox-border"))
    )
    return self.recaptcha_checkbox_border_element


  def _wait_recaptcha_iframe(self) -> None:
    """Wait recaptcha frame"""
    WebDriverWait(self.driver, self.driver_timeout).until(
          expected_conditions.frame_to_be_available_and_switch_to_it(
        (By.CSS_SELECTOR, 'iframe[title="recaptcha challenge expires in two minutes"]')))


  def _wait_image_tiles(self) -> List[WebElement]:
    """Find image tiles"""
    self.image_tile_elements =  WebDriverWait(self.driver, self.driver_timeout).until(
          expected_conditions.presence_of_all_elements_located(
        (By.CLASS_NAME, 'rc-imageselect-tile')))

    return self.image_tile_elements


  def _find_image_tiles(self) -> List[WebElement]:
    """Find image tiles"""
    self._wait_image_tiles()
    self.image_tile_elements = self.driver.find_elements(By.CLASS_NAME, 'rc-imageselect-tile')

    return self.image_tile_elements


  def _find_instructions(self) -> WebElement:
    """Find instructions"""
    self.instructions = WebDriverWait(self.driver, self.driver_timeout).until(
          expected_conditions.presence_of_element_located(
        (By.CLASS_NAME, 'rc-imageselect-desc-no-canonical')))

    return self.instructions

  def _find_status_message(self) -> WebElement:
    """Find instructions"""
    self.status_message = self.driver.find_element(
      By.CLASS_NAME, 'rc-imageselect-incorrect-response')

    return self.status_message


  def _find_submit_button(self) -> WebElement:
    """Find submit button"""
    self.submit_button_element = self.driver.find_element(By.ID, 'recaptcha-verify-button')
    return self.submit_button_element


  def click_recaptcha_checkbox_border(self) -> None:
    """click recaptcha checkbox border"""
    recaptcha_checkbox_border_element = self._find_recaptcha_checkbox_border_element()
    recaptcha_checkbox_border_element.click()
    self.driver.switch_to.default_content()


  def click_submit_button(self) -> None:
    """Find submit button"""
    self.submit_button_element = self._find_submit_button()

    try:
      if self.submit_button_element:
        self.submit_button_element.click()

    except Exception:
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)


  def _get_ai_website_description(self) -> List[SeleniumWebElement]:
    """_get_ai_website_description"""

    base64_image = self.driver.get_screenshot_as_base64()
    explainer_chain = Base64ImageStringExplainerChain(
      model_name = self.model_describer_name,
      max_tokens = self.model_describer_max_tokens,
      prompt = (
        "Your description must be detailed enough"
        " so that you are able to discern between, text, link, form amd button elements")
    )

    result_dict: Dict[str, Any] = explainer_chain.invoke(
      { "base64_image_string": base64_image, "source": self.driver.current_url })

    explainer_result: Base64ImageStringExplainedImage = result_dict["result"]

    self.text_elements = [
      SeleniumWebElement.from_objs(
        element_id = "description",
        description = explainer_result.explanation,
        element_type = CONTENT_KEY
      )
    ]

    return self.text_elements


  def _get_website_main_content(self) -> str:
    """ get website main content """

    self.text_elements = self._get_ai_website_description()

    description = ' '.join([i.description for i in self.text_elements])
    logger.info("Image tiles description: %s", description)
    return description

  def _get_recaptcha_label(self):
    self.recaptcha_label = self.driver.find_element(By.TAG_NAME, 'strong').text
    return self.recaptcha_label


  def _get_selected_image_tile_idxs(self) -> List[int]:
    """get selected list"""
    self.selected_image_tile_idxs = []

    for i, image_tile_element in enumerate(self.image_tile_elements):

      if 'rc-imageselect-tileselected' in image_tile_element.get_attribute('class'):
        self.selected_image_tile_idxs.append(i)

    return self.selected_image_tile_idxs


  def _get_image_tile_selection_by_screenshot(self) -> List[int]:
    self.image_tile_selections = []
    self.image_tile_elements = self._find_image_tiles()
    self.base64_image = self.create_image_element_base64()
    tile_number = len(self.image_tile_elements)
    grid_number = int(sqrt(tile_number))
    self.recaptcha_label = self._get_recaptcha_label().upper()

    self.instructions = self._find_instructions()
    self.status_message = self._find_status_message()

    image_scene_type = ""
    image_behaviour = ""
    selected_segments = ""
    deselect_instructions = ""
    end_message = ""

    self.selected_image_tile_idxs = self._get_selected_image_tile_idxs()
    logger.info("Image tiles selected: %s", self.image_tile_selections)
    selected_image_tile_idxs_str = ','.join([str(i) for i in self.selected_image_tile_idxs])

    if grid_number < 4:
      image_scene_type = " containing a different image scene in each segment,"
      image_behaviour = (
        " refresh the segment with a new image."
        " White tiles are not selectable")
      end_message =  f" if there are no more {self.recaptcha_label} segments."
    else:
      image_scene_type = " separates one image scene,"
      deselect_instructions = (
        " If you want to de-select an already selected segment in the array,"
        f" because it does not contain a {self.recaptcha_label},"
        " assign the segment with a negative index value."
        " For example, to deselect segment 2, the value is -2")
      image_behaviour = " have a blue tick in the top left corner of the selected segment."
      selected_segments = (
        " Currently, the image segments selected"
        f" are:\n{selected_image_tile_idxs_str}\n")
      end_message =  f" if all the {self.recaptcha_label} segments have been selected."

    prompt = (
      " The image contains separating grid lines, creating square segments that"
      f"{image_scene_type}"
      f" Are there any segment that contain {self.recaptcha_label}?"
      " The square segments are represented"
      " with corresponding numbered indexes in red starting with"
      f" 0 and ending with {tile_number - 1} on the image."
      f" Select the segments that contain"
      f" part or all of {self.recaptcha_label} within the boundaries of the segment."
      " by returning a comma delimited indexes."
      " Once a segment is selected, the reCAPTCHA will"
      f" {image_behaviour}"
      f"\n{selected_segments}"
      f"{deselect_instructions}"
      f" Return the string DONE, {end_message}"
      "ONLY Return either a comma delimited indexes or DONE.")

    explainer_chain = Base64ImageStringExplainerChain(
      model_name = self.model_describer_name,
      temperature = self.temperature,
      max_tokens = self.model_describer_max_tokens,
      prompt = prompt
    )

    try:
      result_dict: Dict[str, Any] = explainer_chain.invoke(
        { "base64_image_string": self.base64_image, "source": self.driver.current_url })

      explainer_result: Base64ImageStringExplainedImage = result_dict["result"]

      selection_array_str = explainer_result.explanation
      selection_array_str = selection_array_str.replace(' ', '')

      if re.match(r'^-?\d+(?:,-?\d+)*$', selection_array_str):
        self.image_tile_selections = [
          int(tile_str) for tile_str in selection_array_str.split(',') if tile_str]

      logger.info("image_tile_selections: %s", self.image_tile_selections)

    except Exception:
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()
      self.image_tile_selections = []

    return self.image_tile_selections


  def _enhance_image_brightness(self, input_image: Image, enhance_factor: int) -> Image:
    """Enhance brightness"""
    image_filter = ImageEnhance.Brightness(input_image)
    input_image = image_filter.enhance(enhance_factor)
    return input_image


  def _enhance_image_contrast(self, input_image: Image, enhance_factor: int) -> Image:
    """Enhance contrast"""
    image_filter = ImageEnhance.Contrast(input_image)
    input_image = image_filter.enhance(enhance_factor)
    return input_image


  def _enhance_image_sharpness(self, input_image: Image, enhance_factor: int) -> Image:
    """Enhance sharpness"""
    image_filter = ImageEnhance.Sharpness(input_image)
    input_image = image_filter.enhance(enhance_factor)
    return input_image


  def _enhance_image(self, input_image: Image) -> Image:
    """Enhance image"""
    input_image = self._enhance_image_brightness(input_image, self.brightness)
    input_image = self._enhance_image_contrast(input_image, self.contrast)
    input_image = self._enhance_image_sharpness(input_image, self.sharpness)
    return input_image


  def _add_tile_numbers_to_image(self, input_image: Image, tile_number: int) -> Image:
    """Add numbers to image"""
    cols = int(sqrt(tile_number))
    rows = cols
    length = input_image.size[0]
    width = input_image.size[1]
    tile_length = length / cols
    tile_width = width / rows

    width_offset = 10
    image_draw = ImageDraw.Draw(input_image)
    x = 0
    y = 0

    for i in range(rows):
      y = int(tile_width * i + width_offset)
      for j in range(cols):
        x = int(tile_length / 2 + tile_length * j)
        tile_idx = j + (i * cols)
        image_draw.text((x, y), f"{tile_idx}", fill=(255, 0, 0, 255))

    return input_image


  def _convert_image_to_base64(self, input_image: Image) -> str:
    """_convert_image_to_base64"""
    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


  def _get_image(self) -> WebElement:
    self.image_element = WebDriverWait(self.driver, self.driver_timeout).until(
      expected_conditions.presence_of_element_located(
        (By.XPATH,  "//div[starts-with(@id, 'rc-imageselect-target')]")))
    return self.image_element


  def create_image_element_base64(self) -> str:
    """create_image_element_base64"""
    self.image_element = self._get_image()
    raw_image = Image.open(BytesIO(self.image_element.screenshot_as_png))
    tile_number = len(self.image_tile_elements)
    numbered_image = self._add_tile_numbers_to_image(raw_image, tile_number)
    enhanced_image = self._enhance_image(numbered_image)
    self.base64_image = self._convert_image_to_base64(enhanced_image)
    return self.base64_image


  def _get_image_tile_selection(self) -> List[int]:
    """_get_image_tile_selection"""
    self.image_tile_selections: List[int] = []
    if len(self.image_tile_elements) <= 9:
      time.sleep(random.uniform(self.low_loading_images_delay, self.high_loading_images_delay))
    else:
      time.sleep(
        random.uniform(self.low_loading_image_delay, self.high_loading_image_delay))
    self.image_tile_selections = self._get_image_tile_selection_by_screenshot()
    return self.image_tile_selections


  def click_image_tiles(self) -> bool:
    """select image tiles"""
    change_flag = False
    self.image_tile_selections = self._get_image_tile_selection()

    image_tile_selections = list(set(self.image_tile_selections))
    image_tile_selections.sort()
    for i in image_tile_selections:
      if abs(i) < len(self.image_tile_elements):
        image_tile_element = self.image_tile_elements[abs(i)]
        if i >= 0:
          if 'rc-imageselect-tileselected' not in \
            image_tile_element.get_attribute('class'):
            time.sleep(random.uniform(self.low_click_delay, self.high_click_delay))
            image_tile_element.click()
            change_flag = True
        else:
          if 'rc-imageselect-tileselected' in \
            image_tile_element.get_attribute('class'):
            time.sleep(random.uniform(self.low_click_delay, self.high_click_delay))
            image_tile_element.click()
            change_flag = True
    return change_flag


  def _get_selected_image_tile_idxs(self) -> List[int]:
    """get selected list"""
    self.selected_image_tile_idxs = []

    for i, image_tile_element in enumerate(self.image_tile_elements):

      if 'rc-imageselect-tileselected' in image_tile_element.get_attribute('class'):
        self.selected_image_tile_idxs.append(i)

    return self.selected_image_tile_idxs


  def process_recaptcha(self) -> str:
    """process recaptcha"""
    result_msg = None
    try:
      self.click_recaptcha_checkbox_border()
      self._wait_recaptcha_iframe()
      while True:
        if not self._wait_image_tiles():
          logger.warning("Did not find image tiles")
          break
        change_flag = self.click_image_tiles()

        if len(self.image_tile_elements) < 10:
          change_flag = bool(random.randint(0, 1))
          if not change_flag:
            time.sleep(
              random.uniform(self.low_loading_images_delay, self.high_loading_images_delay))

        logger.info("Click Submit button: %s", not change_flag)

        if not change_flag:
          self.click_submit_button()

    except Exception:
      result_msg = "No reCAPTCHA detected on the website"
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

    try:
      self.driver.switch_to.default_content()
      self.not_robot_iframe_element = self._find_not_robot_iframe()
      self.driver.switch_to.frame(self.not_robot_iframe_element)

      try:
        self.driver.find_element(By.CSS_SELECTOR, "span.recaptcha-checkbox-checked")
        result_msg = "reCAPTCHA solved."
      except Exception:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        filename = ex_traceback.tb_frame.f_code.co_filename
        line_number = ex_traceback.tb_lineno
        logger.error(
          "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
          ex_type, ex_value, ex_traceback, filename, line_number)
        traceback.print_exc()

      if not result_msg:
        try:
          self.driver.find_element(By.CSS_SELECTOR, "span.rc-anchor-error-msg")
          result_msg = "reCAPTCHA not solved."
        except Exception:
          ex_type, ex_value, ex_traceback = sys.exc_info()
          filename = ex_traceback.tb_frame.f_code.co_filename
          line_number = ex_traceback.tb_lineno
          logger.error(
            "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
            ex_type, ex_value, ex_traceback, filename, line_number)
          traceback.print_exc()
          result_msg = "No reCAPTCHA detected on the website"

    except Exception:
      result_msg = "No reCAPTCHA detected on the website"
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

    return result_msg


  @classmethod
  def from_parameters(
    cls,
    driver: WebDriver,
    driver_timeout: Optional[int] = 20,
    model_describer_name: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    model_describer_max_tokens: Optional[int] = 1024,
    tile_by_tile_inference_flag: Optional[bool] = False,
    brightness: Optional[float] = 1.0,
    contrast: Optional[float] = 1.0,
    sharpness: Optional[float] = 1.0):
    """from parameters"""

    return cls(
      driver = driver,
      driver_timeout = driver_timeout,
      model_describer_name = model_describer_name,
      temperature = temperature,
      model_describer_max_tokens = model_describer_max_tokens,
      tile_by_tile_inference_flag = tile_by_tile_inference_flag,
      brightness = brightness,
      contrast = contrast,
      sharpness = sharpness
    )


class DescribeWebsiteInput(BaseModel):
  """Describe website input model."""
  url: str = Field(
    ...,
    description="full URL starting with http or https",
    example="https://www.google.com/",
  )

class ClickButtonInput(BaseModel):
  """Click button input model."""
  element_id: str = Field(
    ...,
    description="element_id of the button you want to click",
    example=(
      " The element_id is 47A11322CA4813853E0638EEBBF08170_element_590"
      " for the following web element json.\n"
      "{  \"element_id\": \"47A11322CA4813853E0638EEBBF08170_element_590\","
      "   \"description\": \"aud list your property register sign in\"  }"),
  )

class ClickLinkInput(BaseModel):
  """Click link input model."""
  element_id: str = Field(
    ...,
    description="element_id of the link you want to click",
    example=(
      " The element_id is 47A11322CA4813853E0638EEBBF08170_element_590"
      " for the following web element json.\n"
      "{  \"element_id\": \"47A11322CA4813853E0638EEBBF08170_element_590\","
      "   \"description\": \"aud list your property register sign in\"  }"),
  )

class FindButtonsInput(BaseModel):
  """Find form input input model."""

  url: Optional[str] = Field(
    default=None,
    description=(
      "Optional argument - the website url."
      " If left empty, the tool will retrieve information"
      " from the website it is currently connected to."),
    example="https://www.google.com/",
  )

class FindLinksInput(BaseModel):
  """Find form input input model."""

  url: Optional[str] = Field(
    default=None,
    description=(
      "Optional argument - the website url."
      " If left empty, the tool will retrieve information"
      " from the website it is currently connected to."),
    example="https://www.google.com/",
  )

class FindFormInputsInput(BaseModel):
  """Find form input input model."""

  url: Optional[str] = Field(
    default=None,
    description=(
      "Optional argument - the website url."
      " If left empty, the tool will retrieve information"
      " from the website it is currently connected to."),
    example="https://www.google.com/",
  )

class FindInteractableElementsInput(BaseModel):
  """Find InteractableElements model."""
  url: Optional[str] = Field(
    default=None,
    description=(
      "Optional argument - the website url."
      " If left empty, the tool will retrieve information"
      " from the website it is currently connected to."),
    example="https://www.google.com/",
  )

class FillOutFormInputsInput(BaseModel):
  """Fill out form input model."""
  form_input: Optional[str] = Field(
    default=None,
    description=(
      "String with the input element_id and their values."
      " The format is element_id1=element_value1|element_id2=element_value2"
      " If there are multiple form input elements, they are delimited by | \""
      ),
    example=(
      "47A11322CA4813853E0638EEBBF08170_element_590=foo@bar.com|"
      "47A11322CA4813853E0638EEBBF08170_element_591=25"
    )
  )

class ScrollInput(BaseModel):
  """Scroll window."""
  direction: str = Field(
      default="down", description="direction to scroll, either 'up' or 'down'"
  )

class SolveRecaptchaInput(BaseModel):
  """Find form input input model."""

  url: Optional[str] = Field(
    default=None,
    description=(
      "Optional argument - the website url."
      " If left empty, the tool will retrieve information"
      " from the website it is currently connected to."),
    example="https://www.google.com/",
  )