# visionlabsai_tast_facedetect
В данном задании необходимо разработать небольшой сервис детекции лиц на изображениях. 
Алгоритм детекции выбирается на усмотрение кандидата, можно воспользоваться любым доступным в open-source.
Решение должно быть масштабируемым на случай высокой нагрузки, на каждое изображение должен быть получен ответ (пусть и с задержкой)
Web-интерфейс должен иметь следующие возможности:
1. Загрузка изображений, после которой автоматически отобразятся ограничивающие
прямоугольники для всех лиц, которые были на нём найдены (возможно с задержкой)
2. Просмотр истории загруженных изображений с выделенными на них ограничивающими прямоугольниками. Для упрощения задания, будем считать, что все пользователи могут
видеть все изображения, которые были загружены в систему

Для первого запуска проекта необходимо запустить 
`docker-compose up --build`

Web на Flask максимально простой и интуитивно понятный

<img width="233" alt="image" src="https://user-images.githubusercontent.com/41343563/169516028-15699936-fed1-4f97-b05e-2a429e0d9125.png">

Результат работы выглядит так

<img width="686" alt="image" src="https://user-images.githubusercontent.com/41343563/169516378-552798cf-dd38-48de-8cdc-2b85bd48d7ae.png">

В galary можно посомтреть историю загруженных изображений 

<img width="1366" alt="image" src="https://user-images.githubusercontent.com/41343563/169516525-a1181493-d603-4703-880f-039aadc4d740.png">

