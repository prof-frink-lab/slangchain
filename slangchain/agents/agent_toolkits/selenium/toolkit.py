""" Selenium Toolkit """
from typing import List

from slangchain.tools.selenium.tool import (
  DescribeWebsiteInput,
  ClickLinkInput,
  FindFormInputsInput,
  ClickButtonInput,
  FillOutFormInputsInput,
  ScrollInput,
  SolveRecaptchaInput,
  SeleniumWrapper
)
from langchain.pydantic_v1 import Field
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.agents import Tool

DESCRIBE_WEBSITE_TOOL_NAME = "Web_Browser_Website_Describer"
PREV_WEBPAGE_NAV_TOOL_NAME = "Web_Browser_Previous_Webpage_Navigator"
FIND_BUTTONS_TOOL_NAME = "Web_Browser_Website_Buttons_Finder"
FIND_LINKS_TOOL_NAME = "Web_Browser_Website_Links_Finder"
FIND_FORMS_TOOL_NAME = "Web_Browser_Website_Form_Inputs_Finder"
FIND_INTERACTABLE_ELEMENTS_TOOL_NAME = "Web_Browser_Interactable_Elements_Finder"
CLICK_BUTTON_TOOL_NAME = "Web_Browser_Website_Button_Clicker"
CLICK_LINK_TOOL_NAME = "Web_Browser_Website_Link_Clicker"
FILL_FORMS_TOOL_NAME = "Web_Browser_Website_Form_Inputs_Filler"
SCROLL_TOOL_NAME = "Web_Browser_Website_Scroller"
SOLVE_RECAPTCHA_TOOL_NAME = "Web_Browser_Recaptcha_Solver"


class SeleniumToolkit(BaseToolkit):
  """Toolkit for interacting with web pages using Selenium ."""

  selenium_wrapper: SeleniumWrapper = Field(exclude=True)

  class Config:
    """Configuration for this pydantic object."""
    arbitrary_types_allowed = True

  def get_tools(self) -> List[BaseTool]:
    """Get the tools in the toolkit."""

    website_describer = Tool(
      name=DESCRIBE_WEBSITE_TOOL_NAME,
      func=self.selenium_wrapper.describe_website,
      description=(
        "Website describer tool"
        " that uses a web browser"
        " to describes web contents on a webpage."
        f" or {FIND_BUTTONS_TOOL_NAME} to get button elements"
        f" Use {FIND_LINKS_TOOL_NAME} to get link elements"
        f" Use {FIND_FORMS_TOOL_NAME} to get form input elements"
        " on the website."
      ),
      args_schema=DescribeWebsiteInput,
    )

    buttons_finder = Tool(
      name=FIND_BUTTONS_TOOL_NAME,
      func=self.selenium_wrapper.find_buttons,
      description=(
        "Website button finder tool that is"
        " that uses a web browser"
        " to find buttons on a website"
        " so that you can interact with the them."
        " The output information can be used by the following tools"
        " to interact with a website:"
        f" {CLICK_BUTTON_TOOL_NAME}."
      ),
    )

    button_clicker = Tool(
      name=CLICK_BUTTON_TOOL_NAME,
      func=self.selenium_wrapper.click_button_by_element_id,
      description=(
        "Website button clicker tool that is"
        " that uses a web browser"
        " to click a button on a website"
        " based on its element_id."
        " The button elements can be found using the outputs from"
        f" {FIND_BUTTONS_TOOL_NAME}."
        " The tool will return the results of the action and a decsription of the website"
        " if there are changes the content on the website after the action."
      ),
      args_schema=ClickButtonInput,
    )

    links_finder = Tool(
      name=FIND_LINKS_TOOL_NAME,
      func=self.selenium_wrapper.find_links,
      description=(
        "Website links finder tool"
        " that uses a web browser"
        " to find links on a website"
        " so that you can interact with the them."
        " The information can be used by the following tools"
        " to interact with a website:"
        f" {CLICK_LINK_TOOL_NAME}."
      ),
    )

    link_clicker = Tool(
      name=CLICK_LINK_TOOL_NAME,
      func=self.selenium_wrapper.click_link_by_element_id,
      description=(
        "Website link clicker tool"
        " that uses a web browser"
        " to click a link on a website"
        " based on its element_id."
        " The link elements can be found using the outputs from"
        f" {FIND_LINKS_TOOL_NAME}."
        " The tool will return the results of the action and a decsription of the website"
        " if there are changes the content on the website after the action."
      ),
      args_schema=ClickLinkInput,
    )

    forms_finder = Tool(
      name=FIND_FORMS_TOOL_NAME,
      func=self.selenium_wrapper.find_forms,
      description=(
        "Website form input finder tool"
        " that uses a web browser"
        " to find form inputs on a website"
        " so that you can interact with the them."
        " The information can be used by the following tools"
        " to interact with a website:"
        f" {FILL_FORMS_TOOL_NAME}."
      ),
      args_schema=FindFormInputsInput,
    )

    form_filler = Tool(
      name=FILL_FORMS_TOOL_NAME,
      func=self.selenium_wrapper.fill_out_form,
      description=(
        "Website form input filler tool"
        " that uses a web browser"
        " to add text to a form input on a website."
        " based on its element_id."
        " The form input elements can be found using the outputs"
        f" of {FIND_FORMS_TOOL_NAME}"
      ),
      args_schema=FillOutFormInputsInput,
    )

    window_scroller = Tool(
      name=SCROLL_TOOL_NAME,
      func=self.selenium_wrapper.scroll,
      description=(
        "Website scroll tool"
        " that uses a web browser"
        " to scroll up or down on a website."
      ),
      args_schema=ScrollInput,
    )

    previous_webpage_navigator = Tool(
      name=PREV_WEBPAGE_NAV_TOOL_NAME,
      func=self.selenium_wrapper.previous_webpage,
      description=(
        "Previous Webpage navigator tool"
        " that uses a web browser go to the previous page"
        " and describes web elements on that page."
        f" Use {FIND_BUTTONS_TOOL_NAME} to get buttons,"
        f" {FIND_LINKS_TOOL_NAME} to get links,"
        f" and {FIND_FORMS_TOOL_NAME} to get inputs forms"
        " on the website."
      ),
    )

    recaptcha_solver = Tool(
      name=SOLVE_RECAPTCHA_TOOL_NAME,
      func=self.selenium_wrapper.solve_recaptcha,
      description=(
        "Website reCAPTCHA solver tool"
        " that uses a web browser"
        " to bypass a reCAPTCHA."),
      args_schema=SolveRecaptchaInput
    )

    return [
      website_describer,
      buttons_finder,
      links_finder,
      forms_finder,
      button_clicker,
      link_clicker,
      form_filler,
      window_scroller,
      previous_webpage_navigator,
      recaptcha_solver
    ]
