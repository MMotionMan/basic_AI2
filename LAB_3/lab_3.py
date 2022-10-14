import pyowm
import telebot
import json
import random

token = '5706181346:AAH5x6uRL3RxhUdjzDN998yhAvWnk2ZOi4U'
bot = telebot.TeleBot(token)
with open('facts.json') as f:
    facts = json.load(f)


def get_location(lat, lon):
    url = f"https://yandex.ru/pogoda/maps/nowcast?lat={lat}&lon={lon}&via=hnav&le_Lightning=1"
    return url


def weather(city: str):
    owm = pyowm.OWM('bed4363a34565fc32ca5bc2e78d2ed25')
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place(city)
    weather = observation.weather
    location = get_location(observation.location.lat, observation.location.lon)
    temperature = weather.temperature("celsius")
    return temperature, location


@bot.message_handler(commands=['start'])
def hello_message(message):
    bot.send_message(message.from_user.id,
                     'Здравствуйте! Я могу рассказать о погоде в зависимости от заданного региона! \n'
                     'Чтобы узнать погоду, используйте команду /weather.')


@bot.message_handler(commands=['weather'])
def get_text_messages(message):
    if message.text == '/weather':
        bot.send_message(message.from_user.id, "Введите название города")
        bot.register_next_step_handler(message, get_weather)
    else:
        bot.send_message(message.from_user.id, 'Напишите /weather')


def get_weather(message):
    city = message.text
    try:
        w = weather(city)
        bot.send_message(message.from_user.id, f'В городе {city} сейчас {round(w[0]["temp"])} градусов,'
                                               f' чувствуется как {round(w[0]["feels_like"])}. \n'
                                               f'А вы знали? {facts["facts"][random.randint(0, len(facts["facts"]))]}\n'
                                               f'{facts["have_a_good_day"][random.randint(0, len(facts["have_a_good_day"]))]}'
                         )
        bot.send_message(message.from_user.id, w[1])
        bot.send_message(message.from_user.id, "Введите название города")
        bot.register_next_step_handler(message, get_weather)
    except Exception:
        bot.send_message(message.from_user.id, 'Извините, такого города нет в базе, попробуйте еще раз')
        bot.send_message(message.from_user.id, "Введите название города")
        bot.register_next_step_handler(message, get_weather)


def main():
    bot.polling(none_stop=True)


while True:
    try:
        main()
    except:
        print('Restarting')
