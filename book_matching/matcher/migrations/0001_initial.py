# Generated by Django 3.2.6 on 2023-12-08 11:45

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('conscientiousness', models.FloatField()),
                ('openness', models.FloatField()),
                ('predicted_genre', models.CharField(max_length=50)),
            ],
        ),
    ]
