from src.ParentParser import ParentParser
from src.ChildParser import ChildParser
from src.Logger import Logger
from src.utils.tools import random_sleep
import random

from css_selectors.livedoor import features, url, subpage, selectors, url_changer


if url[-1] is not '/':
    url += '/'
site_name = url.split('/')[-2]

log = Logger(site_name)

parent = ParentParser(url, subpage, selectors['parent'], log)


for child_url in parent.url_list:
    child_url = url_changer(child_url)
    # If site uses extent link as hyperlink format
    if features['extent_link']:
        if child_url[0] is '/':  # Modify url format
            child_url = child_url[1:]
        child_url = url + child_url
    # Check if duplicated
    if log.check_url(child_url, child=True):
        print('skipped one')
        continue
    # Parse summary if have
    ChildParser(child_url,
                selectors['child'],
                'output/{}/'.format(site_name),
                features['summary_exist'])

    log.log(child_url)

    if random.random() > 0.95:
        print('Long sleep triggered')
        random_sleep(70, 120)
    else:
        random_sleep(15, 50)
